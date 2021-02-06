import importlib
import configparser
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, monotonically_increasing_id
from pyspark.sql.functions import hour, dayofmonth, weekofyear, month, year, date_format
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, DateType, TimestampType
from pyspark.context import SparkContext
from datetime import datetime as dt

def process_song_data(spark, data_song, data_output, key, write_limit = 5):
    '''
    Process the songs data files from S3 path to create and save songs table and artists table.
    Parameters:
        spark       (SparkSession) : Spark session;
        data_song   (str)          : Location of songs data files;
        data_output (str)          : Target for output data files;
        key         (dict)         : Dictionary of output data prefixes;
        write_limit (int)          : Limits records on write for test purpose, 0 is no limit.
    '''    
    # define schema for song data
    song_schema = StructType([
        StructField('song_id', StringType()),
        StructField('artist_id', StringType()),
        StructField('artist_name', StringType()),
        StructField('artist_location', StringType()),
        StructField('artist_latitude', DoubleType()),
        StructField('artist_longitude', DoubleType()),
        StructField('duration', DoubleType()),
        StructField('num_songs', IntegerType()),
        StructField('title', StringType()),
        StructField('year', IntegerType())
    ])
    
    print(f'Reading songs data from: {data_song}')
    
    df = spark.read.json(data_song, schema = song_schema)
    df.cache()
    print(f'Songs records read: {df.count()}')
    
    print('Songs table ingesting.')
    
    global songs_table # for reuse in process_log_data()
    songs_cols = ['song_id', 'title', 'artist_id', 'year', 'duration']
    songs_table = df.select(songs_cols).dropDuplicates(['song_id']).repartition('year', 'artist_id')
    
    print(f'Songs count: {songs_table.count()}')
    
    print('Songs table writing to parquet files partitioned by year and artist.')
    songs_path = data_output + key['songs']
    print(f'Songs location: {songs_path}')
    
    if write_limit > 0:
        print(f'Songs write limit: {write_limit}')
        songs_table.limit(write_limit).write.partitionBy('year', 'artist_id').parquet(songs_path)
    else:
        songs_table.write.partitionBy('year', 'artist_id').parquet(songs_path)

    print('Artists table ingesting.')
    
    artists_cols = ['artist_id',
                    'artist_name as name',
                    'artist_location as location',
                    'artist_latitude as latitude',
                    'artist_longitude as longitude']

    global artists_table # for reuse in process_log_data()
    artists_table = df.selectExpr(artists_cols).dropDuplicates(['artist_id', 'name'])
    
    print(f'Artists count: {artists_table.count()}')
    
    print('Artists table writing to parquet files.')
    artists_path = data_output + key['artists']
    print(f'Artists location: {artists_path}')
    
    if write_limit > 0:
        print(f'Artists write limit: {write_limit}')
        artists_table.limit(write_limit).write.parquet(artists_path)
    else:
        artists_table.write.parquet(artists_path)


def process_log_data(spark, data_log, data_output, key, write_limit = 5):
    '''
    Process the log data files from S3 path to create and save users, time and songplays tables.
    Parameters:
        spark       (SparkSession) : Spark session;
        data_log    (str)          : Location of log data files;
        data_output (str)          : Target for output data files;
        key         (dict)         : Dictionary of output data prefixes;
        write_limit (int)          : Limits records on write for test purpose, 0 is no limit.
    '''
    print(f'Reading log data from: {data_log}')
    
    df = spark.read.json(data_log).where("page = 'NextSong'")
    df.cache()
    print(f"'NextSong' events count: {df.count()}")
    
    print('Users table ingesting.')
    
    users_cols = ['userId as user_id', 'firstName as first_name', 'lastName as last_name', 'gender', 'level']
    users_table = df.selectExpr(users_cols).dropDuplicates(['user_id'])
    
    print(f'Users count: {users_table.count()}')
    
    print('Users table writing to parquet files.')
    users_path = data_output + key['users']
    print(f'Users location: {users_path}')

    if write_limit > 0:
        print(f'Users write limit: {write_limit}')
        users_table.limit(write_limit).write.parquet(users_path)
    else:
        users_table.write.parquet(users_path)
    
    print('Time table ingesting.')

    to_timestamp = udf(lambda x : dt.utcfromtimestamp(x / 1e3), TimestampType())
    df = df.withColumn('start_time', to_timestamp('ts'))

    time_table = df.select('start_time').dropDuplicates() \
        .withColumn('hour', hour('start_time')) \
        .withColumn('day',  dayofmonth('start_time')) \
        .withColumn('week', weekofyear('start_time')) \
        .withColumn('month', month('start_time')) \
        .withColumn('year', year('start_time')) \
        .withColumn('weekday', date_format('start_time', 'E')) \
        .repartition('year', 'month')
    
    print(f'Time stamps count: {time_table.count()}')
    
    print('Time table writing to parquet files partitioned by year and month.')
    time_path = data_output + key['time']
    print(f'Time location: {time_path}')
    
    if write_limit > 0:
        print(f'Time write limit: {write_limit}')
        time_table.limit(write_limit).write.partitionBy('year', 'month').parquet(time_path)
    else:
        time_table.write.partitionBy('year', 'month').parquet(time_path)
    
    print('Songplays table ingesting.')
    
    global songs_table # reused from process_song_data()
    global artists_table # reused from process_song_data()
    
    # select specific columns to skip ambiguous ones
    df = df.join(songs_table.select('song_id', 'title'), (df.song == songs_table.title)) \
           .join(artists_table.select('artist_id', 'name'), (df.artist == artists_table.name))

    # songplay_id will be an autoincrementing column
    # year and month will be added based on start_time
    songplays_cols = ['start_time',
                      'userId as user_id',
                      'level',
                      'song_id',
                      'artist_id',
                      'sessionId as session_id',
                      'location',
                      'userAgent as user_agent']

    songplays_table = df.selectExpr(songplays_cols) \
                        .dropDuplicates(['start_time', 'user_id', 'session_id']) \
                        .withColumn('songplay_id', monotonically_increasing_id()) \
                        .withColumn('month', month('start_time')) \
                        .withColumn('year', year('start_time')) \
                        .repartition('year', 'month')
    
    print(f'Songplays count: {songplays_table.count()}')

    print('Songplays table writing to parquet files partitioned by year and month.')
    songplay_path = data_output + key['songplays']
    print(f'Songplays location: {songplay_path}')
    
    if write_limit > 0:
        print(f'Songplays write limit: {write_limit}')    
        songplays_table.limit(write_limit).write.partitionBy('year', 'month').parquet(songplay_path)
    else:
        songplays_table.write.partitionBy('year', 'month').parquet(songplay_path)


def remove_old_data(bucket, key):
    '''
    Removing the previous output data from the bucket.
    Parameters:
        bucket (boto3.Bucket) : S3 bucket with previous output data;
        key    (dict)         : Dictionary of output data prefixes.
    '''
    print('Removing the previous output data from the bucket.')
    obj_removed = 0
    for obj in list(bucket.objects.filter(Prefix = 'output/songs/')) \
             + list(bucket.objects.filter(Prefix = 'output/artists/')) \
             + list(bucket.objects.filter(Prefix = 'output/users/')) \
             + list(bucket.objects.filter(Prefix = 'output/time/')) \
             + list(bucket.objects.filter(Prefix = 'output/songplays/')):
        _ = obj.delete()
        obj_removed += 1
    print(f'Removed {obj_removed} files.')


def main():
    '''
    Orchestrates the entire ETL process.
    '''
    print('ETL start.')
    
    spark = SparkSession.builder.appName('Sparkify_ETL').getOrCreate()
    print('ETL Spark session open.')
    
    # install module 'boto3' if not present
    if importlib.util.find_spec('boto3') is None:
        print('ETL installs boto3...')
        spark.sparkContext.install_pypi_package('boto3')
    
    import boto3
    
    # read the config file from s3
    bucket = boto3.resource('s3').Bucket('adzugaiev-sparkify')
    dl_cfg = bucket.Object('input/dl.cfg').get()
    
    cfg = configparser.ConfigParser()
    cfg.read_string(dl_cfg['Body'].read().decode())

    data_song = cfg.get('DATA', 'data_song')
    data_log = cfg.get('DATA', 'data_log')
    data_output = cfg.get('DATA', 'data_output')

    key = {
        'songs'    : cfg.get('KEY', 'key_songs'),
        'artists'  : cfg.get('KEY', 'key_artists'),
        'users'    : cfg.get('KEY', 'key_users'),
        'time'     : cfg.get('KEY', 'key_time'),
        'songplays': cfg.get('KEY', 'key_songplays')
    }
    
    remove_old_data(bucket, key)
    process_song_data(spark, data_song, data_output, key) # write_limit = 5
    process_log_data(spark, data_log, data_output, key)  # write_limit = 5
    
    print('ETL complete.')
    spark.stop()


if __name__ == '__main__':
    main()
