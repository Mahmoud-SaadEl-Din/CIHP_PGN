import mysql.connector
from mysql.connector import Error
import pandas as pd
from datetime import date


create_images_table = """
CREATE TABLE images (
  image_id INT AUTO_INCREMENT PRIMARY KEY,
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
CREATE TABLE clothes (
  cloth_id INT AUTO_INCREMENT PRIMARY KEY,
  cloth_path VARCHAR(40) NOT NULL,
  register_time DATE
  );
 """
create_vitons_table = """
CREATE TABLE vitons (
  viton_id INT AUTO_INCREMENT PRIMARY KEY,
  image_id INT NOT NULL,
  cloth_id INT NOT NULL,
  viton_path VARCHAR(40) NOT NULL,
  register_time DATE
  );
 """


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


def show_tables(connection,table):
    cursor = connection.cursor()

    cursor.execute(f"SELECT * FROM {table}")

    # Fetch all rows
    rows = cursor.fetchall()

    # Print the results
    for row in rows:
        print(row)

def insert_rows(connection, table_name, columns, data):

    cursor = connection.cursor()

    # Prepare SQL query
    query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(['%s']*len(columns))})"

    # Execute SQL query for each row in the data
    for row in data:
        cursor.execute(query, row)

    # Commit changes and close connection
    connection.commit()

def delete_all_rows(conn, table_name):

    cursor = conn.cursor()

    # Prepare and execute DELETE query
    delete_query = f"DELETE FROM {table_name}"
    cursor.execute(delete_query)

    # Commit changes and close connection
    conn.commit()

    # cursor.close()


# Example usage
images_table_name = 'images'
clothes_table_name = 'clothes'
vitons_table_name = 'vitons'

images_table_columns = ["img_path","parser_path","pose_path","keypoint_path","seg_agnos_path","gray_agnos_path","densepose_path","register_time"]
clothes_table_columns = ["cloth_id","cloth_path","register_time"]
vitons_table_columns = ["viton_id","image_id","cloth_id","viton_path","register_time"]
data = [
    ('/root/folder/file1.txt', '/root/folder/file2.txt', '/root/folder/file3.txt', '/root/folder/file4.txt', '/root/folder/file5.txt', '/root/folder/file6.txt', '/root/folder/file7.txt', date(2022, 1, 15)),
    ('/user/documents/doc1.txt', '/user/documents/doc2.txt', '/user/documents/doc3.txt', '/user/documents/doc4.txt', '/user/documents/doc5.txt', '/user/documents/doc6.txt', '/user/documents/doc7.txt', date(2024, 1, 15)),
    ('/user/documents/doc1.txt', '/user/documents/doc2.txt', '/user/documents/doc3.txt', '/user/documents/doc4.txt', '/user/documents/doc5.txt', '/user/documents/doc6.txt', '/user/documents/doc7.txt', date(2022, 1, 15)),
    ('/user/documents/doc1.txt', '/user/documents/doc2.txt', '/user/documents/doc3.txt', '/user/documents/doc4.txt', '/user/documents/doc5.txt', '/user/documents/doc6.txt', '/user/documents/doc7.txt', date(2023, 1, 15)),
    # Add more rows as needed
]


my_queries = {
    "create": "CREATE DATABASE VITON",
    "show_images": "SELECT * FROM images",
    "show_clothes": "SELECT * FROM clothes",
    "show_vitons": "SELECT * FROM vitons"
}

def setup_DB():

    created = True
    if not created:
        DB = create_server_connection("localhost","root","Inno26489*")
        create_database(DB, my_queries["create"])
    else:
        DB = create_db_connection("localhost","root","Inno26489*","VITON")
        execute_query(DB, create_images_table)
        execute_query(DB, create_clothes_table)
        execute_query(DB, create_vitons_table)
    return DB

# conn = setup_DB()#create_db_connection("localhost","root","Inno26489*","VITON")#
# insert_rows(conn, table_name, columns, data)
# show_tables(conn, my_queries["show_images"])
# delete_all_rows(conn,"images")
# show_tables(conn, my_queries["show_images"])

# cursor.close()
# connection.close()