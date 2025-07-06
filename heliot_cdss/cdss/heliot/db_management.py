import tiledb
import pandas as pd
import time
from itertools import product
import gc
from unidecode import unidecode
import numpy as np
from typing import List, Dict, Any, Optional
import traceback
from .base_db_mgt import *

class DatabaseManagement(BaseDatabaseManagement):
    #def __init__(self, db_uri="drugs_db", store_lower_case=False, compress_attrs=False, tiles=None):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.db_uri = config.get('db_uri','drugs_db')
        self.store_lower_case = config.get('store_lower_case', False)
        self.compress_attrs = config.get('compress_attrs',False)
        self.tiles = config.get('tiles', None)


    # Transform the potential utf-8 text into ascii unidecode
    def utf8_to_ascii_unidecode(self, text:str) -> str:
        return unidecode(text)

    # Change \xa0 to ' ' into the given text
    def clean_text(self, text:str) -> str:
        return text.replace('\xa0', ' ').strip()

    def create_DBSchema(self):
        # TileDB dimensions. Note: dimensions cannot have dtype=str, it is unsupported.
        drug_code_dim = tiledb.Dim(name="drug_code", tile=self.tiles, dtype="ascii")
        atc_dim = tiledb.Dim(name="atc", tile=self.tiles, dtype="ascii") 
        composition_dim = tiledb.Dim(name="composition", tile=self.tiles, dtype="ascii")
        excipients_dim = tiledb.Dim(name="excipients", tile=self.tiles, dtype="ascii")

        # Create the tileDB domain 
        domain = tiledb.Domain(drug_code_dim, atc_dim, composition_dim, excipients_dim)

        if self.compress_attrs:
            attrs = [
                tiledb.Attr(name="drug_name", dtype=str),
                tiledb.Attr(name="drug_form", dtype=str),
                tiledb.Attr(name="therapeutic_indications", dtype=str, filters=tiledb.FilterList([tiledb.ZstdFilter(level=3)])),
                tiledb.Attr(name="posology", dtype=str,filters=tiledb.FilterList([tiledb.ZstdFilter(level=3)])),
                tiledb.Attr(name="cross_reactivity", dtype=str, filters=tiledb.FilterList([tiledb.ZstdFilter(level=3)])),
                tiledb.Attr(name="contraindications", dtype=str,filters=tiledb.FilterList([tiledb.ZstdFilter(level=3)])),
                tiledb.Attr(name="special_warnings", dtype=str, filters=tiledb.FilterList([tiledb.ZstdFilter(level=3)])),
                tiledb.Attr(name="drug_interactions", dtype=str, filters=tiledb.FilterList([tiledb.ZstdFilter(level=3)])),
                tiledb.Attr(name="pregnancy_info", dtype=str, filters=tiledb.FilterList([tiledb.ZstdFilter(level=3)])),
                tiledb.Attr(name="driving_effects", dtype=str,filters=tiledb.FilterList([tiledb.ZstdFilter(level=3)])),
                tiledb.Attr(name="side_effects", dtype=str,filters=tiledb.FilterList([tiledb.ZstdFilter(level=3)])),
                tiledb.Attr(name="over_dose", dtype=str,filters=tiledb.FilterList([tiledb.ZstdFilter(level=3)])),
                tiledb.Attr(name="incompatibilities", dtype=str,filters=tiledb.FilterList([tiledb.ZstdFilter(level=3)])),
                tiledb.Attr(name="leaflet", dtype=str)
            ]
        else:
            # Attribute definition
            attrs = [
                tiledb.Attr(name="drug_name", dtype=str),
                tiledb.Attr(name="drug_form", dtype=str),
                tiledb.Attr(name="therapeutic_indications", dtype=str),
                tiledb.Attr(name="posology", dtype=str),
                tiledb.Attr(name="cross_reactivity", dtype=str),
                tiledb.Attr(name="contraindications", dtype=str),
                tiledb.Attr(name="special_warnings", dtype=str),
                tiledb.Attr(name="drug_interactions", dtype=str),
                tiledb.Attr(name="pregnancy_info", dtype=str),
                tiledb.Attr(name="driving_effects", dtype=str),
                tiledb.Attr(name="side_effects", dtype=str),
                tiledb.Attr(name="over_dose", dtype=str),
                tiledb.Attr(name="incompatibilities", dtype=str),
                tiledb.Attr(name="leaflet", dtype=str)
            ]

        # Create the array schema
        schema = tiledb.ArraySchema(
            domain=domain,
            attrs=attrs,
            sparse=True  # As per TileDB documentation, we set it as Sparse, because the dimensions are variable strings
        )

        # Create the tileDB array
        tiledb.Array.create(self.db_uri, schema)

    def to_lower_case(self, value):
        return value.lower() if isinstance(value, str) else value

    # Insert a single drug
    def insert_drug(self, drug_code: str, drug_name: str, atc: str, drug_form: str, composition: List, excipients: List, therapeutic_indications: str, posology: str, cross_reactivity: str, contraindications: str, special_warnings: str, drug_interactions: str, pregnancy_info: str, driving_effects: str, side_effects: str, over_dose: str, incompatibilities: str, leaflet:str) -> bool:
        dtype_dict = {
            "drug_code": str,
            "drug_name": str,
            "atc": str,
            "drug_form": str,
            "therapeutic_indications": str,
            "posology": str,
            "cross_reactivity": str,
            "contraindications": str,
            "special_warnings": str,
            "drug_interactions": str,
            "pregnancy_info": str,
            "driving_effects": str,
            "side_effects": str,
            "over_dose": str,
            "incompatibilities": str,
            "leaflet": str
        }
        data = {key: [] for key in dtype_dict.keys() if key != "composition" and key != "excipients"}
        data["composition"] = []
        data["excipients"] = []

        if self.store_lower_case:
            composition = [self.to_lower_case(comp) for comp in composition]
            excipients = [self.to_lower_case(exc) for exc in excipients]
            drug_name = self.to_lower_case(drug_name)
            atc = self.to_lower_case(atc)
            drug_form = self.to_lower_case(drug_form)
            therapeutic_indications = self.to_lower_case(therapeutic_indications)
            posology = self.to_lower_case(posology)
            cross_reactivity = self.to_lower_case(cross_reactivity)
            contraindications = self.to_lower_case(contraindications)
            special_warnings = self.to_lower_case(special_warnings)
            drug_interactions = self.to_lower_case(drug_interactions)
            pregnancy_info = self.to_lower_case(pregnancy_info)
            driving_effects = self.to_lower_case(driving_effects)
            side_effects = self.to_lower_case(side_effects)
            over_dose = self.to_lower_case(over_dose)
            incompatibilities = self.to_lower_case(incompatibilities)
            leaflet = self.to_lower_case(leaflet)

        product_list = list(product(composition, excipients))
        data["drug_code"].extend([drug_code] * len(product_list))
        data["atc"].extend([atc] * len(product_list))
        data["composition"].extend([comp for comp, exc in product_list])
        data["excipients"].extend([exc for comp, exc in product_list])
        data["drug_name"].extend([drug_name] * len(product_list))
        data["drug_form"].extend([drug_form] * len(product_list))
        data["therapeutic_indications"].extend([therapeutic_indications] * len(product_list))
        data["posology"].extend([posology] * len(product_list))
        data["cross_reactivity"].extend([cross_reactivity] * len(product_list))
        data["contraindications"].extend([contraindications] * len(product_list))
        data["special_warnings"].extend([special_warnings] * len(product_list))
        data["drug_interactions"].extend([drug_interactions] * len(product_list))
        data["pregnancy_info"].extend([pregnancy_info] * len(product_list))
        data["driving_effects"].extend([driving_effects] * len(product_list))
        data["side_effects"].extend([side_effects] * len(product_list))
        data["over_dose"].extend([over_dose] * len(product_list))
        data["incompatibilities"].extend([incompatibilities] * len(product_list))
        data["leaflet"].extend([leaflet] * len(product_list))
        unique_data = {}
        for j in range(len(data["drug_code"])):
            key = (data["drug_code"][j], data["atc"][j], data["composition"][j], data["excipients"][j])
            if key not in unique_data:
                unique_data[key] = {attr: data[attr][j] for attr in data}

        # Prepare data for batch insertion
        formatted_coords = {
            "drug_code": [],
            "atc": [],
            "composition": [],
            "excipients": [],
        }
        formatted_data = {attr: [] for attr in data if attr not in ["drug_code", "atc", "composition", "excipients"]}

        # Populate the attributes and dimensions for batch insertion
        for key, value in unique_data.items():
            formatted_coords["drug_code"].append(key[0])
            formatted_coords["atc"].append(key[1])
            formatted_coords["composition"].append(key[2])
            formatted_coords["excipients"].append(key[3])
            for attr in value:
                if attr in formatted_data:
                    formatted_data[attr].append(value[attr])

        # Check the coordinates and attributes sizes just for debugging
        coord_length = len(formatted_coords["drug_code"])
        attr_length = len(next(iter(formatted_data.values())))
        print(f"Coord length: {coord_length}, Attr length: {attr_length}")
        result = True
        with tiledb.open(self.db_uri, mode="w") as array:
            # Sanity check. If it is ok, we can write data
            if coord_length == attr_length:
                array[formatted_coords["drug_code"], formatted_coords["atc"], formatted_coords["composition"], formatted_coords["excipients"]] = {k: np.array(v) for k, v in formatted_data.items()}
            else:
                print("Length mismatch between coordinates and attributes!")
                result = False
        return result

    # Create the TileDB database for drugs
    def create_and_populate_DBFromCSV(self, csv_file='leaflet_info.csv'):
        start_time = time.time()  # Start measurement time

        # Specify the datatypes for the cvs file to read. No dataype is specified for 'composition' and 'excipients'
        # This is why we want to use lists during the insertion phase for 'composition' and 'excipients'
        dtype_dict = {
            "drug_code": str,
            "drug_name": str,
            "atc": str,
            "drug_form": str,
            "therapeutic_indications": str,
            "posology": str,
            "cross_reactivity": str,
            "contraindications": str,
            "special_warnings": str,
            "drug_interactions": str,
            "pregnancy_info": str,
            "driving_effects": str,
            "side_effects": str,
            "over_dose": str,
            "incompatibilities": str,
            "leaflet": str
        }

        # Read the CSV file
        df = pd.read_csv(csv_file, dtype=dtype_dict, encoding='utf-8')

        # Convert all columns into strings and update NaN to ''
        df = df.fillna('').astype(str)
        for col in df.columns:
            df[col] = df[col].apply(self.clean_text)
            if self.store_lower_case:
                df[col] = df[col].apply(self.to_lower_case)

        #check_special_characters(df)

        self.create_DBSchema()

        total_rows = len(df)
        batch_size = 100  # Batch size to insert data into TileDB
        data = {key: [] for key in dtype_dict.keys() if key != "composition" and key != "excipients"}
        data["composition"] = []
        data["excipients"] = []

        with tiledb.open("drugs_db", mode="w") as array:
            for i, row in df.iterrows():
                compositions = [self.utf8_to_ascii_unidecode(x.strip()) for x in row['composition'].split('#')]
                excipients = [self.utf8_to_ascii_unidecode(x.strip()) for x in row['excipients'].split('#')]

                product_list = list(product(compositions, excipients))

                #print(f"Drug code: {row['drug_code']}, Compositions: {compositions}, Excipients: {excipients}, Combinations: {len(product_list)}")

                data["drug_code"].extend([row["drug_code"]] * len(product_list))
                data["composition"].extend([comp for comp, exc in product_list])
                data["excipients"].extend([exc for comp, exc in product_list])
                data["drug_name"].extend([row["drug_name"]] * len(product_list))
                data["atc"].extend([row["atc"]] * len(product_list))
                data["drug_form"].extend([row["drug_form"]] * len(product_list))
                data["therapeutic_indications"].extend([row["therapeutic_indications"]] * len(product_list))
                data["posology"].extend([row["posology"]] * len(product_list))
                data["cross_reactivity"].extend([row["cross_reactivity"]] * len(product_list))
                data["contraindications"].extend([row["contraindications"]] * len(product_list))
                data["special_warnings"].extend([row["special_warnings"]] * len(product_list))
                data["drug_interactions"].extend([row["drug_interactions"]] * len(product_list))
                data["pregnancy_info"].extend([row["pregnancy_info"]] * len(product_list))
                data["driving_effects"].extend([row["driving_effects"]] * len(product_list))
                data["side_effects"].extend([row["side_effects"]] * len(product_list))
                data["over_dose"].extend([row["over_dose"]] * len(product_list))
                data["incompatibilities"].extend([row["incompatibilities"]] * len(product_list))
                data["leaflet"].extend([row["leaflet"]] * len(product_list))

                # Write data when we reach the batch size
                if (i + 1) % batch_size == 0 or (i + 1) == total_rows:
                    print("I", i, len(data))

                    # Remove duplicates. It shouldn't happend
                    unique_data = {}
                    for j in range(len(data["drug_code"])):
                        key = (data["drug_code"][j], data["atc"][j], data["composition"][j], data["excipients"][j])
                        if key not in unique_data:
                            unique_data[key] = {attr: data[attr][j] for attr in data}

                    # Prepare data for batch insertion
                    formatted_coords = {
                        "drug_code": [],
                        "atc": [],
                        "composition": [],
                        "excipients": [],
                    }
                    formatted_data = {attr: [] for attr in data if attr not in ["drug_code", "atc", "composition", "excipients"]}

                    # Populate the attributes and dimensions for batch insertion
                    for key, value in unique_data.items():
                        formatted_coords["drug_code"].append(key[0])
                        formatted_coords["atc"].append(key[1])
                        formatted_coords["composition"].append(key[2])
                        formatted_coords["excipients"].append(key[3])
                        for attr in value:
                            if attr in formatted_data:
                                formatted_data[attr].append(value[attr])

                    # Check the coordinates and attributes sizes just for debugging
                    coord_length = len(formatted_coords["drug_code"])
                    attr_length = len(next(iter(formatted_data.values())))
                    print(f"Coord length: {coord_length}, Attr length: {attr_length}")

                    # Sanity check. It is ok, we can write data
                    if coord_length == attr_length:
                        array[formatted_coords["drug_code"], formatted_coords["atc"], formatted_coords["composition"], formatted_coords["excipients"]] = {k: np.array(v) for k, v in formatted_data.items()}
                    else:
                        print("Length mismatch between coordinates and attributes!")
                    
                    print("End Writing")

                    # Reset Dictionary
                    data = {key: [] for key in dtype_dict.keys() if key != "composition" and key != "excipients"}
                    data["composition"] = []
                    data["excipients"] = []

                    # Enforce garbage collection
                    gc.collect()

                # Print the progress every 100 processed drugs
                if (i + 1) % 100 == 0 or (i + 1) == total_rows:
                    print(f"Processed drugs {i + 1} out of {total_rows}")

        end_time = time.time()  # End duration masurement
        print(f"Execution time for the database creation: {end_time - start_time:.2f} seconds")


    # Search a drug given its code
    def search_drug(self, drug_code:str) -> Dict:
        # Open TileDB in read mode
        with tiledb.open(self.db_uri, mode="r") as array:
            # Conditional query for the given drug_code 
            data = array.query(attrs=["drug_name", "drug_form", "therapeutic_indications", "posology", "cross_reactivity", "contraindications", "special_warnings", "drug_interactions", "pregnancy_info", "driving_effects", "side_effects", "over_dose", "incompatibilities", "leaflet"]).df[drug_code]

            if data.empty:
                return None

            # Aggregate composition and excipients as sets
            compositions = set(data['composition'])
            excipients = set(data['excipients'])

            # Create the dictionary to return
            drug_info = data.iloc[0].to_dict()
            drug_info['composition'] = list(compositions)
            drug_info['excipients'] = list(excipients)

        return drug_info


    def find_by_compositions(self, compositions) -> Dict:
        # Normalize compositions
        if self.store_lower_case:
            encoded_compositions = [self.to_lower_case(self.utf8_to_ascii_unidecode(comp)) for comp in compositions]
        else:
            encoded_compositions = [self.utf8_to_ascii_unidecode(comp) for comp in compositions]

        # Open TileDB in read mode
        with tiledb.open(self.db_uri, mode="r") as array:
            # Filter by composition using a query filter
            condition = " and ".join(["composition == \"\"\""+comp.replace('\"', "'")+"\"\"\"" for comp in encoded_compositions])
            data = array.query(attrs=["drug_name"], cond=condition).multi_index[:, encoded_compositions, :]
            r = pd.DataFrame.from_dict(data)
            return r

    def find_by_compositions_direct_slice(self, composition) ->Dict:

        #if self.store_lower_case:
        #    encoded_compositions = [self.to_lower_case(self.utf8_to_ascii_unidecode(comp)) for comp in composition]
        #else:
        #    encoded_compositions = [self.utf8_to_ascii_unidecode(comp) for comp in composition]

        # Open TileDB in read mode
        with tiledb.open(self.db_uri, mode="r") as array:
            #data = array.multi_index[:,encoded_compositions,:]
            data = array.query(attrs=["drug_name"]).df[:, composition, :]
            #print(type(data))
            #print(data)
            #data = array.query(cond=query)[:]
            return pd.DataFrame.from_dict(data)
            #print(r)
            #return r

        
    # Filter the database by composition. Composition is a list
    def filter_by_composition(self, uri, compositions) ->Dict:
        # Open TileDB in read mode
        with tiledb.open(self.db_uri, mode="r") as array:
            # Create a DataFrame to combine the results for each composition in the list
            dfs = []
            for comp in compositions:
                result = array.query(coords=True).multi_index[:, comp, :]
                df = pd.DataFrame.from_dict(result)
                dfs.append(df)
            combined_df = pd.concat(dfs)
        return combined_df

    # Search drugs that have the given compositions as active ingredients and do not contain the given excipients (one is enough to exclude a drug)
    def search_drugs_by_composition_and_excipients(self, compositions:List, excipients:List) -> Dict:
        #start_time = time.time()  # Start the measurement time

        # Apply ascii unicode conversion 
        if self.store_lower_case:
            encoded_compositions = [self.to_lower_case(self.utf8_to_ascii_unidecode(comp)) for comp in compositions]
            encoded_excipients = [self.to_lower_case(self.utf8_to_ascii_unidecode(exc)) for exc in excipients]
        else:
            encoded_compositions = [self.utf8_to_ascii_unidecode(comp) for comp in compositions]
            encoded_excipients = [self.utf8_to_ascii_unidecode(exc) for exc in excipients]

        # filter the database by active ingredients
        #data_df = self.filter_by_composition("drugs_db", encoded_compositions)
        data_df = self.find_by_compositions_direct_slice(encoded_compositions)        
        #end_time = time.time()  # End of measurement time
        #print(f"Tempo di esecuzione filter_by_composition: {end_time - start_time:.2f} secondi")
        # Debug: Print filtered data for the fiven compositions
        #print("Filtered data for the fiven compositions:")
        #print(data_df.head())

        # Filter the results of the previous query to exclude the drugs that contain at least an excipient in the given list 
        filtered_data = data_df[~data_df['excipients'].isin(encoded_excipients)]

        # Debug: Filtered data for the fiven excipients
        #print("Filtered data for the fiven excipients:")
        #print(filtered_data.head())

        if filtered_data.empty:
            return None

        # Aggregate the result by drug_code
        aggregated_result = []
        grouped_data = filtered_data.groupby('drug_code')
        for drug_code, data in grouped_data:
            compositions = set(data['composition'])
            excipients = set(data['excipients'])

            # Create the dictionary to return
            drug_info = data.iloc[0].to_dict()
            drug_info['composition'] = list(compositions)
            drug_info['excipients'] = list(excipients)

            #CONVERSION
            if isinstance(drug_code, bytes):
                drug_info['drug_code'] = drug_code.decode('utf-8')  # Decode bytes and create a string value
            # Convert byte columns to strings 
            for key in drug_info.keys():
                if isinstance(drug_info[key], bytes):
                    drug_info[key] = drug_info[key].decode('utf-8')
            #END CONVERSION

            aggregated_result.append(drug_info)

        #end_time = time.time()  # End of measurement time
        #print(f"Tempo di esecuzione per la ricerca dei farmaci: {end_time - start_time:.2f} secondi")

        return aggregated_result

    # delete drugs by code
    def delete_drug_by_code(self, drug_code: str) -> bool:
        try:
            # Create the condition to delete rows matching the specified drug_code
            cond = f"drug_code == '{drug_code}'"

            # Issue the delete query with the condition
            with tiledb.open(self.db_uri, mode="d") as array:
                array.query(cond=cond).submit()
                print(f"Deleted rows with drug_code: {drug_code}")
            return True
        except Exception as e:
            print(f"Error deleting rows with drug_code {drug_code}: {e}")
        return False


    # Update specific attributes of a drug by drug_code
    def update_drug(self, drug_code: str, update_data: Dict):
        try:
            # Read the existing data for the given drug_code
            existing_data = self.search_drug(drug_code)
            if not existing_data:
                print(f"No data found for drug_code: {drug_code}")
                return

            # Update only the specified fields in update_data
            for key, value in update_data.items():
                if key in existing_data:
                    if self.store_lower_case and key != "composition" and key != "excipients":
                        existing_data[key] = self.to_lower_case(value)
                    else:
                        existing_data[key] = value

            # Extract necessary details for re-insertion
            composition = existing_data.pop("composition", [])
            excipients = existing_data.pop("excipients", [])
            
            if self.store_lower_case:
                composition = [self.to_lower_case(self.utf8_to_ascii_unidecode(comp)) for comp in composition]
                excipients = [self.to_lower_case(self.utf8_to_ascii_unidecode(comp)) for comp in excipients]

            # Prepare data for updating
            update_coords = {
                "drug_code": [drug_code] * len(composition) * len(excipients),
                "atc": [existing_data['atc']] * len(composition) * len(excipients),
                "composition": [comp for comp in composition for _ in excipients],
                "excipients": [exc for _ in composition for exc in excipients]
            }
            update_attrs = {key: [value] * len(update_coords["drug_code"]) for key, value in existing_data.items()}
            
            # Combine coordinates and attributes
            combined_data = {**update_coords, **update_attrs}
            
            # Perform update/insert
            with tiledb.open(self.db_uri, mode="w") as array:
                array[update_coords["drug_code"], update_coords["atc"], update_coords["composition"], update_coords["excipients"]] = {
                    k: np.array(v) for k, v in combined_data.items() if k not in ["drug_code", "atc", "composition", "excipients"]
                }

            print(f"Updated drug with drug_code: {drug_code}")

        except Exception as e:
            print(f"Error updating drug with drug_code {drug_code}: {e}")

    def count_records(self) -> int:
        # Open TileDB in read mode
        with tiledb.open(self.db_uri, mode="r") as A:
            q = A.query()
    
            # count the number of records in the array
            d = q.agg("count")[:]
            return d["drug_name"]["count"]


    def get_all_drugs(self) -> List[Dict]:
        try:
            # Open TileDB in read mode
            with tiledb.open(self.db_uri, mode="r") as array:
                # Query all data
                data = array.query(attrs=["drug_name", "drug_form", "therapeutic_indications", "posology", "cross_reactivity", "contraindications", "special_warnings", "drug_interactions", "pregnancy_info", "driving_effects", "side_effects", "over_dose", "incompatibilities", "leaflet"]).df[:]
            
            if data.empty:
                return []

            # Aggregate composition and excipients as sets
            aggregated_result = []
            grouped_data = data.groupby('drug_code')
            for drug_code, data in grouped_data:
                compositions = set(data['composition'])
                excipients = set(data['excipients'])

                # Create the dictionary to return
                drug_info = data.iloc[0].to_dict()
                drug_info['composition'] = list(compositions)
                drug_info['excipients'] = list(excipients)

                # Convert byte columns to strings if necessary
                if isinstance(drug_code, bytes):
                    drug_info['drug_code'] = drug_code.decode('utf-8')  # Decode bytes and create a string value
                for key in drug_info.keys():
                    if isinstance(drug_info[key], bytes):
                        drug_info[key] = drug_info[key].decode('utf-8')

                aggregated_result.append(drug_info)

            return aggregated_result

        except Exception as e:
            print(f"Error retrieving all drugs: {e}")
            return []

    def find_drugs_by_atc(self, atc_code: str, drug_code_to_exclude: str = None) -> List[Dict]:
        try:
            # Converti in minuscolo se necessario
            if self.store_lower_case:
                atc_code = self.to_lower_case(atc_code)
            
            print(f"Cercando farmaci con ATC: {atc_code}")

            # Apri TileDB in modalità lettura
            with tiledb.open(self.db_uri, mode="r") as array:
                # Query per il codice ATC specifico
                condition = f"atc == \"\"\"{atc_code}\"\"\""
                print(f"Condizione query: {condition}")
                
                # Prima verifichiamo se ci sono dati con questo ATC
                test_data = array.query(coords=True).df[:, atc_code, :]
                print(f"Dati trovati (test): {len(test_data) if not test_data.empty else 0} righe")

                # Query completa
                data = array.query(attrs=["drug_name", "drug_form", "therapeutic_indications", 
                                        "posology", "cross_reactivity", "contraindications", 
                                        "special_warnings", "drug_interactions", "pregnancy_info", 
                                        "driving_effects", "side_effects", "over_dose", 
                                        "incompatibilities", "leaflet"]).df[:, atc_code, :]
                
                print(f"Dati trovati (query completa): {len(data) if not test_data.empty else 0} righe")

            if data.empty:
                print("Nessun dato trovato")
                return []

            # Aggrega i risultati per drug_code
            aggregated_result = []
            grouped_data = data.groupby('drug_code')
            
            for drug_code, group_data in grouped_data:
                if drug_code_to_exclude and drug_code == drug_code_to_exclude:
                    continue
                compositions = set(group_data['composition'])
                excipients = set(group_data['excipients'])

                # Crea il dizionario per il farmaco
                drug_info = group_data.iloc[0].to_dict()
                drug_info['composition'] = list(compositions)
                drug_info['excipients'] = list(excipients)

                # Converti i bytes in stringhe se necessario
                if isinstance(drug_code, bytes):
                    drug_info['drug_code'] = drug_code.decode('utf-8')
                for key in drug_info.keys():
                    if isinstance(drug_info[key], bytes):
                        drug_info[key] = drug_info[key].decode('utf-8')

                aggregated_result.append(drug_info)

            print(f"Numero di farmaci trovati dopo aggregazione: {len(aggregated_result)}")
            return aggregated_result

        except Exception as e:
            print(f"Errore nella ricerca dei farmaci per codice ATC {atc_code}: {e}")
            traceback.print_exc()  # Stampa lo stack trace completo
            return []

    
if __name__ == "__main__":
    # Create the database
    config = {'store_lower_case':True}
    dm = DatabaseManagement(config) #, compress_attrs=True, tiles=20)
    dm.create_and_populate_DBFromCSV()

    # Search a drug by drug_code 
    drug_code_to_search = "044089062"  # Choose a different drug_code you want to search for
    start_time = time.time()  # Start the measurement time
    result = dm.search_drug(drug_code_to_search)
    end_time = time.time()  # End measurement time
    print(f"Execution time to search the drug by code: {end_time - start_time:.2f} seconds")

    if result:
        print("Drug found", result['atc'], result['cross_reactivity'])
        #print("DRUG CODE",result['drug_code'],"\n", print(result['side_effects']))
    else:
        print("Drug not found")

    # Search for a new drug_code
    start_time = time.time()  # Start the measurement time
    result = dm.search_drug("020766034")
    end_time = time.time()  # End measurement time
    print(f"Execution time to search the drug by code: {end_time - start_time:.2f} seconds")

    if result:
        cmp = result['composition']
        print("DRUG CODE",result['drug_code'], result['atc'])
    print("\n")
    #print(result['side_effects'])


    start_time = time.time()  # Start the measurement time
    result = dm.search_drugs_by_composition_and_excipients(["Aripiprazolo"], ["Sodio"])
    end_time = time.time()  # End measurement time
    print(f"Execution time to search the drug by composition and excipients: {end_time - start_time:.2f} seconds")

    if result is not None:
        for r in result:
            print("=====> ROW")
            print(r['drug_code'])
            print("\n")

    start_time = time.time()  # Start the measurement time
    result = dm.search_drugs_by_composition_and_excipients(["Aripiprazolo"], ["Sodio"])
    end_time = time.time()  # End measurement time
    print(f"Execution time to search the drug by composition and excipients: {end_time - start_time:.2f} seconds")

    '''
    print("UPDATE DRUG")
    start_time = time.time()  # Start the measurement time
    dm.update_drug("027890146",{"drug_name":"pippo"})
    end_time = time.time()  # End measurement time
    print(f"Execution time to update the drug: {end_time - start_time:.2f} seconds")
    '''

    start_time = time.time()  # Start the measurement time
    result = dm.search_drug("027890146")
    end_time = time.time()  # End measurement time
    print(f"Execution time to search the drug by code: {end_time - start_time:.2f} seconds")

    if result is not None:
        print("=====> ROW", result['drug_code'])
        #for r in result:
        #    print("=====> ROW", result[r])
            #print(r['drug_code'])
        #    print("\n")
    else:
        print("DRUG CODE 027890146 not found")


    result = dm.find_drugs_by_atc('A02BA02','035398357')
    if result is not None:
        print("=============== SIMILAR DRUGS")
        for r in result:
            print(r['drug_code'])
        print("=============== ")
    '''
    print("DELETE DRUG")
    start_time = time.time()  # Start the measurement time
    dm.delete_drug_by_code("027890146")
    end_time = time.time()  # End measurement time
    print(f"Execution time to delete the drug by code: {end_time - start_time:.2f} seconds")

    start_time = time.time()  # Start the measurement time
    result = dm.search_drug("027890146")
    end_time = time.time()  # End measurement time
    print(f"Execution time to search the drug by code: {end_time - start_time:.2f} seconds")

    if result is not None:
        print("=====> ROW", result['drug_code'])
        #for r in result:
        #    print("=====> ROW", result[r])
            #print(r['drug_code'])
        #    print("\n")
    else:
        print("DRUG CODE 027890146 not found")

    '''

    print("COUNT TOTAL")
    start_time = time.time()  # Start the measurement time
    print("TOTAL",dm.count_records())
    end_time = time.time()  # End measurement time
    print(f"Execution time to count rows: {end_time - start_time:.2f} seconds")

    start_time = time.time()  # Start the measurement time
    dm.find_by_compositions(cmp)
    end_time = time.time()  # End measurement time
    print(f"Execution time to query compositions: {end_time - start_time:.2f} seconds")


    start_time = time.time()  # Start the measurement time
    dm.find_by_compositions_direct_slice(cmp)
    end_time = time.time()  # End measurement time
    print(f"Execution time to query find_by_compositions_direct_slice: {end_time - start_time:.2f} seconds")


    start_time = time.time()  # Start the measurement time
    drug_code_to_search = "034329110"  # Choose a different drug_code you want to search for
    result = dm.search_drug(drug_code_to_search)
    end_time = time.time()  # End measurement time
    print(f"Execution time to query by drug_code: {end_time - start_time:.2f} seconds")