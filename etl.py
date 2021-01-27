import configparser
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, monotonically_increasing_id
from pyspark.sql.functions import hour, dayofmonth, weekofyear, month, year, date_format
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, DateType, TimestampType
from datetime import datetime as dt

def create_spark_session():
    '''
    Creates and returns the Spark session.
    '''
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.5") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    '''
    Process the songs data files from S3 path to create and save songs table and artists table.
    Parameters:
        spark       (SparkSession) : Spark session;
        input_data  (str)          : Location of songs data files;
        output_data (str)          : Target for songs and artists data files.
    '''    
    # get filepath to song data file
    song_data = input_data + 'song_data/*/*/*/*.json'

    # define schema for song data
    song_schema = StructType([
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
    
    print('Reading songs data from: {}'.format(song_data))
    
    df = spark.read.json(song_data, schema = song_schema)
    df.cache()
    print('Songs records read: {}'.format(df.count()))
    
    print('Songs table ingesting.')
    
    # song_id will be an autoincrementing column
    songs_cols = ['title', 'artist_id', 'year', 'duration']
    global songs_table # for reuse in process_log_data()
    
    songs_table = df.select(songs_cols).dropDuplicates().withColumn('song_id', monotonically_increasing_id())
    print('Songs count: {}'.format(songs_table.count()))
    
    print('Songs table writing to parquet files partitioned by year and artist.')
    print('Songs location: {}'.format(output_data + 'songs/'))
    
    songs_table.write.partitionBy('year', 'artist_id').parquet(output_data + 'songs/') #.limit(5)

    print('Artists table ingesting.')
    
    artists_cols = ['artist_id',
                'artist_name as name',
                'artist_location as location',
                'artist_latitude as latitude',
                'artist_longitude as longitude']

    # using selectExpr() because of 'as' column aliases present
    global artists_table # for reuse in process_log_data()
    
    artists_table = df.selectExpr(artists_cols).dropDuplicates()
    print('Artists count: {}'.format(artists_table.count()))
    
    print('Artists table writing to parquet files.')
    print('Artists location: {}'.format(output_data + 'artists/'))
    
    artists_table.write.parquet(output_data + 'artists/') #.limit(5)


def process_log_data(spark, input_data, output_data):
    '''
    Process the log data files from S3 path to create and save users, time and songplays tables.
    Parameters:
        spark       (SparkSession) : Spark session;
        input_data  (str)          : Location of log data files;
        output_data (str)          : Target for users, time and songplays data files.
    '''
    log_data = input_data + 'log-data/*/*/*.json'
    
    print('Reading log data from: {}'.format(log_data))
    
    df = spark.read.json(log_data).where("page = 'NextSong'")
    df.cache()
    print("'NextSong' events count: {}".format(df.count()))
    
    print('Users table ingesting.')
    
    users_cols = ['userId as user_id', 'firstName as first_name', 'lastName as last_name', 'gender', 'level']
    users_table = df.selectExpr(users_cols).dropDuplicates()
    print('Users count: {}'.format(users_table.count()))
    
    print('Users table writing to parquet files.')
    print('Users location: {}'.format(output_data + 'users/'))
    
    users_table.write.parquet(output_data + 'users/') #.limit(5)
    
    print('Time table ingesting.')

    to_timestamp = udf(lambda x : dt.utcfromtimestamp(x / 1e3), TimestampType())
    df = df.withColumn('start_time', to_timestamp('ts'))

    time_table = df.select('start_time').dropDuplicates() \
        .withColumn('hour', hour('start_time')) \
        .withColumn('day',  dayofmonth('start_time')) \
        .withColumn('week', weekofyear('start_time')) \
        .withColumn('month', month('start_time')) \
        .withColumn('year', year('start_time')) \
        .withColumn('weekday', date_format('start_time', 'E')) # https://stackoverflow.com/a/12781297
    print('Time stamps count: {}'.format(time_table.count()))
    
    print('Time table writing to parquet files partitioned by year and month.')
    print('Time location: {}'.format(output_data + 'time/'))
    
    time_table.write.partitionBy('year', 'month').parquet(output_data + 'time/') #.limit(5)
    
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

    songplays_table = df.selectExpr(songplays_cols).dropDuplicates() \
                        .withColumn('songplay_id', monotonically_increasing_id()) \
                        .withColumn('month', month('start_time')) \
                        .withColumn('year', year('start_time'))

    print('Songplays count: {}'.format(songplays_table.count()))

    print('Songplays table writing to parquet files partitioned by year and month.')
    print('Songplays location: {}'.format(output_data + 'songplays/'))
    
    songplays_table.write.partitionBy('year', 'month').parquet(output_data + 'songplays/')


def main():
    '''
    Orchestrates the entire ETL process.
    '''
    print('ETL start.')
    
    cfg = configparser.ConfigParser()
    cfg.read('dl.cfg')

    os.environ['AWS_ACCESS_KEY_ID'] = cfg.get('AWS', 'AWS_KEY')
    os.environ['AWS_SECRET_ACCESS_KEY'] = cfg.get('AWS', 'AWS_SECRET')
    input_data = cfg.get('DATA', 'DATA_INPUT')
    output_data = cfg.get('DATA', 'DATA_OUTPUT')
    
    spark = create_spark_session()
    print('ETL Spark session open.')
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)
    
    print('ETL complete.')


if __name__ == '__main__':
    main()
