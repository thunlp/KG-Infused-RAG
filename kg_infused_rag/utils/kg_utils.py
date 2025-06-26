import os
import json
from tqdm import tqdm
from collections import defaultdict

import logging
logger = logging.getLogger(__name__)

class KgRetriever:
    def __init__(self, kg_path, corpus_path, entity_alias_path, relation_alias_path, json_dir):
        self.kg_path = kg_path
        self.corpus_path = corpus_path
        self.entity_alias_path = entity_alias_path
        self.relation_alias_path = relation_alias_path
        self.json_dir = json_dir

        # Ensure that the json directory exists
        self.create_directory(self.json_dir)

        # Load data
        print("Loading data...")
        self.kg_data = self.load_or_convert_to_json("triple", kg_path, os.path.join(json_dir, self.get_json_filename(kg_path)))
        self.corpus_data = self.load_or_convert_to_json("text", corpus_path, os.path.join(json_dir, self.get_json_filename(corpus_path)))
        
        # Load or convert entity alias data, and create a mapping from entity name to ID
        entity_name_to_id_path = os.path.join(json_dir, "entity_name_to_id.json")
        relation_name_to_id_path = os.path.join(json_dir, "relation_name_to_id.json")
        
        self.entity_alias_data = self.load_or_convert_to_json("entity", entity_alias_path, os.path.join(json_dir, self.get_json_filename(entity_alias_path)))   # [{Q5196650: ['Cut Your Hair', 'cut your hair']}, {...} ...]
        self.entity_alias_data_reverse = self.transform_dicts(self.entity_alias_data)

        self.relation_alias_data = self.load_or_convert_to_json("relation", relation_alias_path, os.path.join(json_dir, self.get_json_filename(relation_alias_path)))
        self.relation_alias_data_reverse = self.transform_dicts(self.relation_alias_data)

        self.entity_alias_data_mapping = self.reverse_entity_mapping(self.entity_alias_data, entity_name_to_id_path)
        self.relation_alias_data_mapping = self.reverse_entity_mapping(self.relation_alias_data, relation_name_to_id_path)

        print("Data loading complete!")


    def create_directory(self, path):
        """Create directory if it does not exist."""
        if not os.path.exists(path):
            print(f"Directory {path} does not exist. Creating it...")
            os.makedirs(path)
            print(f"Directory {path} created successfully.")
        else:
            print(f"Directory {path} already exists.")

    def load_or_convert_to_json(self, data_source, file_path, json_path):
        """Load data from file or convert it to JSON and save."""
        if os.path.exists(json_path):
            print(f"Found existing JSON file: {json_path}. Loading...")
            with open(json_path, "r") as f:
                data = json.load(f)
            print(f"{json_path} loaded successfully!")
            return data
        else:
            print(f"JSON file not found. Converting data from {file_path} to JSON...")
            data = {}
            with open(file_path, "r") as f:
                total_lines = sum(1 for _ in f)  # Get the total number of lines in the file for tqdm
            with open(file_path, "r") as f:
                for line in tqdm(f, total=total_lines, desc=f"Processing {file_path}", mininterval=1.0):
                    parts = line.strip().split("\t")
                    if data_source == "triple":
                        data.setdefault(parts[0], []).append((parts[1], parts[2]))
                    elif data_source in ["entity", "relation", "text"]:
                        data[parts[0]] = parts[1:]
                    else:
                        raise Exception("data source not supported.")
            with open(json_path, "w") as f:
                json.dump(data, f)
            print(f"Conversion complete. Data saved to {json_path}!")
            return data
    
    @staticmethod
    def transform_dicts(list_of_dict):
        """Transforms a list of dictionaries by swapping keys and values, 
        where values are lists of strings.
        """
        result = {}
        for key, values  in list_of_dict.items():
            for value in values:
                result[value] = key
        return result

    def reverse_entity_mapping(self, data, output_file_path):
        """
        Converts a mapping from entity ID to entity names into a mapping from entity names to entity IDs,
        and saves the result as a JSON file.
        """
        if os.path.exists(output_file_path):
            print(f"File {output_file_path} found. Loading the existing reversed mapping...")
            with open(output_file_path, 'r', encoding='utf-8') as f:
                entity_name_to_ids = json.load(f)
            print("Mapping loaded successfully!")
            return entity_name_to_ids

        # Process the data to create a reverse mapping
        print("Processing entity mappings...")
        entity_name_to_ids = defaultdict(list)
        for entity_id, name_list in data.items():
            for name in name_list:
                name = name.lower()
                entity_name_to_ids[name]= entity_id
        
        # Save the reverse mapping as a JSON file
        print(f"Saving reversed mapping to {output_file_path}...")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(entity_name_to_ids, f, ensure_ascii=False, indent=4)
        print("Mapping saved successfully!")
        return entity_name_to_ids

    def get_json_filename(self, file_path):
        """Generate a JSON filename by replacing the file extension with .json."""
        filename = os.path.basename(file_path)
        name, _ = os.path.splitext(filename)  # Remove the original extension
        return f"{name}.json"

    def get_entity_description(self, entity_name):
        """Given an entity name, return its description."""
        entity_name = entity_name.lower()  # Normalize to lowercase
        entity_id = self.entity_alias_data_mapping.get(entity_name)
        if entity_id:
            return self.corpus_data.get(entity_id, "Description not found")
        else:
            return "Entity not found"

    def get_entity_names(self, entity_id):
        """Given an entity id, return its entity name."""
        return self.entity_alias_data.get(entity_id, ["Entity not found"])
    
    def get_entity_names_batch(self, entity_ids):
        """Given a list of entity IDs, return a dictionary mapping each ID to its entity names."""
        return {entity_id: self.entity_alias_data.get(entity_id, ["Entity not found"]) for entity_id in entity_ids}
    
    def get_entity_id(self, entity_name):
        """Given an entity name, return its entity id."""
        return self.entity_alias_data_reverse.get(entity_name, ["Entity not found"])

    def get_relation_names(self, relation_id):
        """Given an relationship id, return its relationship name."""
        return self.relation_alias_data.get(relation_id, ["Relationship not found"])

    def get_relation_id(self, relation_name):
        """Given an relationship name, return its relationship id."""
        return self.relation_alias_data_reverse.get(relation_name, ["Relationship not found"])

    def get_entity_triples_by_id(self, entity_id, return_subject=True, trans_to_text=True):
        """Given an entity name, return its triples"""
        triples = []
        if trans_to_text:
            if return_subject:
                entity_name = self.get_entity_names(entity_id)[0]   # select first entity name
                triples.extend([(
                    entity_name, 
                    self.get_relation_names(relation)[0], 
                    self.get_entity_names(target)[0]) for relation, target in self.kg_data.get(entity_id, [])])
            else:
                triples.extend([(
                    self.get_relation_names(relation)[0], 
                    self.get_entity_names(target)[0]) for relation, target in self.kg_data.get(entity_id, [])])
        else:
            if return_subject:
                triples.extend([(entity_id, relation, target) for relation, target in self.kg_data.get(entity_id, [])])
            else:
                triples.extend([(relation, target) for relation, target in self.kg_data.get(entity_id, [])])
        return triples

    def get_entity_triples_by_id_batch(self, entity_ids, return_subject=False, trans_to_text=True):
        """
        Given a list of entity IDs, return a dictionary mapping each ID to its triples.
        """
        results = {}

        for entity_id in entity_ids:
            triples = []
            entity_name = self.get_entity_names(entity_id)[0] if trans_to_text and return_subject else None

            if trans_to_text:
                if return_subject:
                    triples = [(
                        entity_name, 
                        self.get_relation_names(relation)[0], 
                        self.get_entity_names(target)[0]
                    ) for relation, target in self.kg_data.get(entity_id, [])]
                else:
                    triples = [(
                        self.get_relation_names(relation)[0], 
                        self.get_entity_names(target)[0]
                    ) for relation, target in self.kg_data.get(entity_id, [])]
            else:
                if return_subject:
                    triples = [(entity_id, relation, target) for relation, target in self.kg_data.get(entity_id, [])]
                else:
                    triples = [(relation, target) for relation, target in self.kg_data.get(entity_id, [])]

            results[entity_id] = triples

        return results


    def get_entity_triples(self, entity_name, return_subject=True, trans_to_text=True):
        """Given an entity name, return its triples"""
        entity_name_lower = entity_name.lower()  # Normalize to lowercase
        entity_id = self.entity_alias_data_mapping.get(entity_name_lower)
        if entity_id:
            triples = []
            if trans_to_text:
                if return_subject:
                    triples.extend([(
                        entity_name, 
                        self.get_relation_names(relation)[0], 
                        self.get_entity_names(target)[0]) for relation, target in self.kg_data.get(entity_id, [])])
                else:
                    triples.extend([(
                        self.get_relation_names(relation)[0], 
                        self.get_entity_names(target)[0]) for relation, target in self.kg_data.get(entity_id, [])])
            else:
                if return_subject:
                    triples.extend([(entity_id, relation, target) for relation, target in self.kg_data.get(entity_id, [])])
                else:
                    triples.extend([(relation, target) for relation, target in self.kg_data.get(entity_id, [])])
            return triples
        else:
            return None

    def get_entity_relations(self, entity_name, trans_to_text=True):
        """Given an entity name, return its relationships"""
        entity_name = entity_name.lower()  # Normalize to lowercase
        entity_id = self.entity_alias_data_mapping.get(entity_name)
        if entity_id:
            relations = []
            if trans_to_text:
                relations.extend([(
                    entity_name, 
                    self.get_relation_names(relation)[0]) for relation, _ in self.kg_data.get(entity_id, [])])
            else:
                relations.extend([(entity_id, relation) for relation, _ in self.kg_data.get(entity_id, [])])
            return relations
        else:
            return None


class KgLoader:
    @classmethod
    def load_wikidata5m(cls, path):
        """
        Loads the KG of Wikidata5M.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Error: The file '{path}' does not exist.")
        
        logger.info(f"Loading knowledge from: {path}")
        entity_knowledge = []

        with open(path) as f:
            if path.endswith(".json"):
                entity_knowledge = json.load(f)
            elif path.endswith(".jsonl"):
                entity_knowledge = [json.loads(line) for line in f]
            else:
                raise ValueError("Unsupported file format. Expected .json or .jsonl")
        
        return entity_knowledge
