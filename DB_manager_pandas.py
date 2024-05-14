import pandas as pd
import os
from os.path import join
from os.path import exists



class DB:
    def __init__(self):
        # Define paths to CSV files
        self.db_folder = 'DB'
        self.cloth_csv = join(self.db_folder, 'cloth.csv')
        self.persons_csv = join(self.db_folder, 'persons.csv')
        self.vitons_csv = join(self.db_folder, 'vitons.csv')
        
        self.cloth_columns = ['id', 'cloth_path', 'register_time']
        self.persons_columns = ['id', 'img_path', 'register_time']
        self.vitons_columns = ['id', 'image_id', 'cloth_id', 'viton_path', 'register_time']
        
        self.cloth_df = self._load_or_create_dataframe(self.cloth_csv, self.cloth_columns)
        self.persons_df = self._load_or_create_dataframe(self.persons_csv, self.persons_columns)
        self.vitons_df = self._load_or_create_dataframe(self.vitons_csv, self.vitons_columns)
        
        self.last_cloth_id = DB._get_last_id(self.cloth_df)
        self.last_persons_id = DB._get_last_id(self.persons_df)
        self.last_vitons_id = DB._get_last_id(self.vitons_df)
        
    @staticmethod    
    def _get_last_id(df):
        # Check if DataFrame is empty
        if df.empty:
            return 0  # Start with ID 0 if DataFrame is empty
        else:
            # Get the maximum ID value from the DataFrame
            last_id =df['id'].max()
            return last_id
    def _load_or_create_dataframe(self, path, columns):
        """Load a dataframe from a CSV file. Create the file if it doesn't exist."""
        if not exists(path):
                df = pd.DataFrame(columns=columns)
                df.to_csv(path, index=False)
        
        # Load the dataframe from CSV file or return an empty dataframe
        return pd.read_csv(path)

    def read_all_rows(self, dataframe_name):
        """Read all rows from a specified dataframe."""
        if dataframe_name == 'cloth':
            return self.cloth_df
        elif dataframe_name == 'persons':
            return self.persons_df
        elif dataframe_name == 'vitons':
            return self.vitons_df
        else:
            return pd.DataFrame()

    def add_row(self, dataframe_name, row_data):
        """Add a new row to a specified dataframe."""
        id_added = 0
        print("in the function ", row_data)
        if dataframe_name == 'cloth':
            self.last_cloth_id +=1
            r = [self.last_cloth_id]
            r.extend(row_data)
            df = pd.DataFrame(columns=self.cloth_columns, data=[r])
            self.cloth_df = pd.concat([self.cloth_df,df],axis=0)
            self.cloth_df.to_csv(self.cloth_csv, index=False)
            return self.last_cloth_id
        elif dataframe_name == 'persons':
            self.last_persons_id +=1
            r = [self.last_persons_id]
            r.extend(row_data)
            df = pd.DataFrame(columns=self.persons_columns, data=[r])
            self.persons_df = pd.concat([self.persons_df,df],axis=0)
            self.persons_df.to_csv(self.persons_csv, index=False)
            return self.last_persons_id
        elif dataframe_name == 'vitons':
            self.last_vitons_id +=1
            r = [self.last_vitons_id]
            r.extend(row_data)
            df = pd.DataFrame(columns=self.vitons_columns, data=[r])
            self.vitons_df = pd.concat([self.vitons_df,df],axis=0)
            self.vitons_df.to_csv(self.vitons_csv, index=False)
            return self.last_vitons_id
        else:
            print("not added anything")
            return -1

    def delete_row(self, dataframe_name, index):
        """Delete a row from a specified dataframe by index."""
        if dataframe_name == 'cloth':
            if self.cloth_df is not None:
                self.cloth_df = self.cloth_df.drop(index, axis=0)
        elif dataframe_name == 'persons':
            if self.persons_df is not None:
                self.persons_df = self.persons_df.drop(index, axis=0)
        elif dataframe_name == 'vitons':
            if self.vitons_df is not None:
                self.vitons_df = self.vitons_df.drop(index, axis=0)
        else:
            raise ValueError("Invalid dataframe name")

    def get_cloth_id_by_name(self, cloth_name):
        """Get cloth_id by cloth_name."""
        filtered_df = self.cloth_df[self.cloth_df['cloth_path'] == cloth_name]
        if not filtered_df.empty:
            return filtered_df['id'].iloc[0]
        else:
            print(f"No cloth found with name: {cloth_name}")
            return None

    def get_image_id_by_name(self, image_name):
        """Get image_id by img_path."""
        filtered_df = self.persons_df[self.persons_df['img_path'] == image_name]
        if not filtered_df.empty:
            return filtered_df['id'].iloc[0]
        else:
            print(f"No image found with name: {image_name}")
            return None

    def get_cloth_name_by_id(self, cloth_id):
        """Get cloth_path by cloth_id."""
        filtered_df = self.cloth_df[self.cloth_df['id'] == cloth_id]
        if not filtered_df.empty:
            return filtered_df['cloth_path'].iloc[0]
        else:
            print(f"No cloth found with id: {cloth_id}")
            return None

    def get_image_name_by_id(self, image_id):
        """Get img_path by image_id."""
        filtered_df = self.persons_df[self.persons_df['id'] == image_id]
        if not filtered_df.empty:
            return filtered_df['img_path'].iloc[0]
        else:
            print(f"No image found with id: {image_id}")
            return None


# Example usage:
if __name__ == "__main__":
    db = DB()
    # db.load_dataframes()

    # # Read all rows from cloth dataframe
    # cloth_data = db.read_all_rows('cloth')
    # print("Cloth Data:")
    # print(cloth_data)

    # # Add a new row to the persons dataframe
    # new_person = {'Name': 'John', 'Age': 30}
    # db.add_row('persons', new_person)

    # # Read all rows from persons dataframe after adding a row
    # persons_data = db.read_all_rows('persons')
    # print("\nPersons Data:")
    # print(persons_data)

    # # Delete a row from vitons dataframe
    # db.delete_row('vitons', 0)

    # # Read all rows from vitons dataframe after deleting a row
    # vitons_data = db.read_all_rows('vitons')
    # print("\nVitons Data:")
    # print(vitons_data)
