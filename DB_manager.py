import mysql.connector
from mysql.connector import Error
import pandas as pd

def create_server_connection(host_name, user_name, user_password):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password
        )
        print("MySQL Database connection successful")
    except Error as err:
        print(f"Error: '{err}'")

    return connection


def create_db_connection(host_name, user_name, user_password, db_name):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name
        )
        print("MySQL Database connection successful")
    except Error as err:
        print(f"Error: '{err}'")

    return connection


def create_database(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        print("Database created successfully")
    except Error as err:
        print(f"Error: '{err}'")

def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query successful")
    except Error as err:
        print(f"Error: '{err}'")


my_queries = {
    "create": "CREATE DATABASE VITON"
}


DB = create_server_connection("localhost","Saad","Encro")
# DB = create_db_connection("localhost","Saad","Encro","VITON")

create_database(DB, my_queries["create"])

create_images_table = """
CREATE TABLE teacher (
  image_id INT PRIMARY KEY,
  img_path VARCHAR(40) NOT NULL,
  parser_path VARCHAR(40),
  pose_path VARCHAR(40),
  keypoint_path VARCHAR(40),
  seg_agnos_path VARCHAR(40),
  gray_agnos_path VARCHAR(40),
  densepose_path VARCHAR(40),
  register_time DATE
  );
 """
create_clothes_table = """
CREATE TABLE teacher (
  cloth_id INT PRIMARY KEY,
  cloth_path VARCHAR(40) NOT NULL,
  register_time DATE
  );
 """

create_vitons_table = """
CREATE TABLE teacher (
  viton_id INT PRIMARY KEY,
  image_id INT NOT NULL,
  cloth_id INT NOT NULL,
  viton_path VARCHAR(40) NOT NULL,
  register_time DATE
  );
 """


execute_query(DB, create_teacher_table) # Execute our defined query
