import mysql.connector
from mysql.connector import Error
from datetime import date


create_images_table = """
CREATE TABLE images (
  image_id INT AUTO_INCREMENT PRIMARY KEY,
  img_path VARCHAR(40) NOT NULL,
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
  viton_path VARCHAR(200) NOT NULL,
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


def show_tables(table):
    connection = create_db_connection("localhost","root","Inno26489*","VITON")
    cursor = connection.cursor()
    print("Table")
    cursor.execute(f"SELECT * FROM {table}")

    # Fetch all rows
    rows = cursor.fetchall()

    # Print the results
    for row in rows:
        print(row)

def insert_rows(table_name, columns, data):
    connection = create_db_connection("localhost","root","Inno26489*","VITON")
    cursor = connection.cursor()

    # Prepare SQL query
    query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(['%s']*len(columns))})"

    # Execute SQL query for each row in the data
    for row in data:
        cursor.execute(query, row)

    # Commit changes and close connection
    connection.commit()

def delete_all_rows(table_name):
    conn = create_db_connection("localhost","root","Inno26489*","VITON")

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

images_table_columns_v1 = ["img_path","parser_path","pose_path","keypoint_path","seg_agnos_path","gray_agnos_path","densepose_path"]
images_table_columns_v2 = ["img_path"]
clothes_table_columns = ["cloth_path"]
vitons_table_columns = ["image_id","cloth_id","viton_path"]

my_queries = {
    "create": "CREATE DATABASE VITON",
    "show_images": "SELECT * FROM images",
    "show_clothes": "SELECT * FROM clothes",
    "show_vitons": "SELECT * FROM vitons"
}


def get_cloth_id_by_name(cloth_name):
    # Replace 'your_username', 'your_password', 'your_database' with your actual credentials
    connection = create_db_connection("localhost","root","Inno26489*","VITON")
    cursor = connection.cursor()

    try:
        # Execute the SELECT query to retrieve cloth_id based on cloth_name
        query = "SELECT cloth_id FROM clothes WHERE cloth_path = %s"
        cursor.execute(query, (cloth_name,))

        # Fetch the result
        result = cursor.fetchone()

        if result:
            cloth_id = result[0]
            return cloth_id
        else:
            print(f"No cloth found with name: {cloth_name}")
            return None

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

    finally:
        cursor.close()
        connection.close()

def get_cloth_name_by_id(id):
    # Replace 'your_username', 'your_password', 'your_database' with your actual credentials
    connection = create_db_connection("localhost","root","Inno26489*","VITON")
    cursor = connection.cursor()

    try:
        # Execute the SELECT query to retrieve cloth_id based on cloth_name
        query = "SELECT cloth_path FROM clothes WHERE cloth_id = %s"
        cursor.execute(query, (id,))

        # Fetch the result
        result = cursor.fetchone()

        if result:
            cloth_id = result[0]
            return cloth_id
        else:
            print(f"No cloth found with id: {id}")
            return None

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

    finally:
        cursor.close()
        connection.close()


def get_image_id_by_name(image_name):
    # Replace 'your_username', 'your_password', 'your_database' with your actual credentials
    connection = create_db_connection("localhost","root","Inno26489*","VITON")
    cursor = connection.cursor()

    try:
        # Execute the SELECT query to retrieve cloth_id based on cloth_name
        query = "SELECT image_id FROM images WHERE img_path = %s"
        cursor.execute(query, (image_name,))

        # Fetch the result
        result = cursor.fetchone()

        if result:
            cloth_id = result[0]
            return cloth_id
        else:
            print(f"No image found with name: {image_name}")
            return None

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

    finally:
        cursor.close()
        connection.close()


def get_image_name_by_id(id):
    # Replace 'your_username', 'your_password', 'your_database' with your actual credentials
    connection = create_db_connection("localhost","root","Inno26489*","VITON")
    cursor = connection.cursor()

    try:
        # Execute the SELECT query to retrieve cloth_id based on cloth_name
        query = "SELECT img_path FROM images WHERE image_id = %s"
        cursor.execute(query, (id,))

        # Fetch the result
        result = cursor.fetchone()

        if result:
            cloth_id = result[0]
            return cloth_id
        else:
            print(f"No image found with id: {id}")
            return None

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

    finally:
        cursor.close()
        connection.close()

def drop_all_tables():
    # Replace 'your_username', 'your_password', 'your_database' with your actual credentials
    connection = create_db_connection("localhost","root","Inno26489*","VITON")

    cursor = connection.cursor()

    try:
        # Get a list of all tables in the database
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()

        # Drop each table
        for table in tables:
            table_name = table[0]
            cursor.execute(f"DROP TABLE {table_name}")

        print("All tables dropped successfully.")

    except mysql.connector.Error as err:
        print(f"Error: {err}")

    finally:
        cursor.close()
        connection.close()

def setup_DB():

    created = True
    if not created:
        DB = create_server_connection("localhost","root","Inno26489*")
        create_database(DB, my_queries["create"])
    else:
        DB = create_db_connection("localhost","root","Inno26489*","VITON")
        drop_all_tables()
        execute_query(DB, create_images_table)
        execute_query(DB, create_clothes_table)
        execute_query(DB, create_vitons_table)
    return DB

# conn = setup_DB()#create_db_connection("localhost","root","Inno26489*","VITON")#
# insert_rows(conn, table_name, columns, data)
# DB = create_db_connection("localhost","root","Inno26489*","VITON")
show_tables(images_table_name)
show_tables(clothes_table_name)
show_tables(vitons_table_name)
# execute_query(DB, "DROP TABLE vitons")
# execute_query(DB, create_vitons_table)

# delete_all_rows("images")
# show_tables(conn, my_queries["show_images"])

# cursor.close()
# connection.close()