from enum import Enum
import os
import json

import numpy as np
import torch
import torch.nn.functional as F

from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer
from plip.plip import PLIP


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

'''
TODO: 
- create json file for embeddings
- finalize types of fields/structures of jsons


TODO implementation: 
- Process embeddings
- Get similarity results
- Compare models
- testing

Model methods - should focus on returning captions (return uuid if image)

initializing
- keys
- captions
- embeddings

interface:
- compare models
- show captions and images

'''

# Enum of model types
class Model_Name(Enum):
    PLIP_TEXT = "PLIP Text"
    PLIP_IMAGE = "PLIP Image"
    PUBMED_BERT = "PubMedBERT"
    BIO_WORD_VEC = "BioWordVec"

# Base Model for creating caption embeddings and calculating/ranking semantic similarity
class Model:
    def __init__(self, name, model, captions_path, images_path, embeddings_json_path, embed_image=False):
        self.name = name
        self.model = model

        print('Parsing captions json...')
        self.keys, self.captions, self.captions_dict = self.parse_caption_json(captions_path)

        self.images = {}

        images = os.listdir(images_path)

        for key in self.captions_dict.keys():
            image = f'{key}.jpg'
            if image in images:
                self.images[key] = f'{images_path}/{image}'
    
        self.embed_image = embed_image

        # print(f'Name: {self.name}')
        # print('Captions dict')
        # print(self.captions_dict)
        # print('keys:')
        # print(self.keys)
        # print('captions:')
        # print(self.captions)

        # print('Processing embeddings...')
        if embeddings_json_path:
            embeddings_dict = self.get_embeddings_json(embeddings_json_path)
            preprocessed_embeddings = [embeddings_dict[key]['embedding'] for key in embeddings_dict.keys()]
        else:
            preprocessed_embeddings = self.preprocess_embeddings()

        
        self.embeddings = preprocessed_embeddings

    # Throws error if file doe not exist
    def get_embeddings_json(self, file_path=None):
        '''
            If there is a json file of embeddings, use those,
            if there is no file, process embeddings and create a json file to refer to incase gets lost
            TODO: have an option for reloading embeddings
        '''

        if file_path is None:
            file_path = f"./preprocessed_embeddings/{self.name}_embeddings.json" # finalize path for json

        with open(file_path, "r") as f:
            embeddings_json = json.load(f)
            # Convert embeddings to np.array
        for key in embeddings_json.keys():
            embeddings_json[key]["embedding"] = np.array(embeddings_json[key]["embedding"])

        return embeddings_json
            
    # only with certain formatting
    def parse_caption_json(self, captions_path):
        with open(captions_path, 'r') as f:
            raw_data = json.load(f)
            # TODO: parse json to get unified code
        
        parsed_data = {}

        for key in raw_data.keys(): 
            parsed_data[raw_data[key]["uuid"]] = raw_data[key]["caption"]

        keys = [raw_data[key]['uuid'] for key in raw_data.keys()]
        captions = [parsed_data[key] for key in parsed_data.keys()]
        return keys, captions, parsed_data
    
    def get_image_paths(self, images_dir_path):
        return [f"{images_dir_path}/{image}" for image in os.listdir(images_dir_path)]
    
    # Returns the cosine similarity score of two text
    def getSimilarityScore(self, text1, text2):
        embedding1 = self.getEmbedding(text1)
        embedding2 = self. getEmbedding(text2)

        return F.cosine_similarity(torch.tensor(embedding1), torch.tensor(embedding2))
    
    # returns top n captions and data of all similarity scores
    def getSimilarityScores(self, query, n=5):
        print('getting similarity scores...')
        query_embedding = self.getEmbedding(query)
        similarity_scores = [F.cosine_similarity(torch.tensor(query_embedding), torch.tensor([array])) for array in self.embeddings]

        results = {}

        scores, indices = torch.topk(torch.Tensor(similarity_scores), k = n)
        for i in range(len(scores)):
            key = self.keys[indices[i]]
        
            results[i+1] = {
                "uuid": key,
                "caption": self.captions_dict[key],
                "similarity_score": scores[i]
            }

        # TODO: option to return all similarity scores 

        return results

    # Prints queried images and captions to users
    def showResults(self, results):
        for key in results.keys():
            uuid = results[key]['uuid']
            sim_score = results[key]['similarity_score']
            print(f'{key}.\nUUID: {uuid}\nCaption{self.captions_dict[uuid]}\nEval Score: {sim_score}')
            if uuid in self.images.keys():
                img = mpimg.imread(self.images[uuid])
                plt.imshow(img)
                plt.axis('off')
                plt.show()

class PLIP_Embedder(Model):
    '''
    data_path: String --> path to the captions.json file or path to the pathology images directory
    '''
    def __init__(self, captions_path, images_path, embeddings_json_path=None, embed_images = False):

        print('Loading PLIP model...')
        model = PLIP('vinid/plip')
        print('model loaded.')

        if embed_images:
            name = Model_Name.PLIP_IMAGE
        else:
            name = Model_Name.PLIP_TEXT

        

        super().__init__(name, model, captions_path, images_path, embeddings_json_path, embed_images)

    def preprocess_embeddings(self):
        """
        Computes the embedding of all the captions in this object.

        :return: List of embeddings.
        """
        if self.embed_images:    
            return self.getAllEmbeddings(self.images, True)
        else:
            return self.getAllEmbeddings(self.captions)
    
    def getAllEmbeddings(self, data_list, embed_images=False):
        """
        Computes the embeddings of a list of data (captions or images) using the PLIP model.

        :return: List of embeddings.
        """
        if embed_images:
            embeddings = self.model.encode_images(data_list, batch_size=32)
        else:
            embeddings = self.model.encode_text(data_list, batch_size=32)

        embeddings = embeddings/np.linalg.norm(embeddings, ord=2, axis=-1, keepdims=True)

        return embeddings
    
    # TODO: Implement to be flexible with images
    def getEmbedding(self, text, embed_image = False):
        """
        Computes the embeddings of a singular text.

        :return: Embedding of type list[float]
        """
        return self.getAllEmbeddings([text], embed_image)
        

class BioWordVec_Embedder(Model):
    def __init__(self, model_path, captions_path, images_path, embeddings_json_path=None):
        """
        Initializes an object that calculates semantic similarity using the BioWordVec Model.
        
        model_path: Path to the BioWordVec_PubMed_MIMICIII_d200.vec.bin file
        captions_path: Path to the captions json file.
        images_path: Path to the images folder.
        embeddings_json_path (optional): Path to precomputed json_path.
        """

        print("loading BioWordVec model...")

        #TODO: figure out what to do if the model is not loaded
        try:
            biowordvec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
            print("model loaded.")
        except:
            # TODO: throw error
            raise RuntimeError("Error: model already loaded. If error persists, restart kernel")

        self.name = Model_Name.BIO_WORD_VEC

        super().__init__(Model_Name.BIO_WORD_VEC, biowordvec_model, captions_path, images_path, embeddings_json_path)

    def preprocess_embeddings(self):
        return self.getAllEmbeddings(self.captions, False)

    def getAllEmbeddings(self, data_list, embed_images):

        if embed_images:
            print("Warning: BioWordVec is unable to process images. Processing captions instead...")
        
        embeddings = [self.getEmbedding(caption) for caption in data_list]

        # embeddings = embeddings/np.linalg.norm(embeddings, ord=2, axis=-1, keepdims=True)

        return embeddings
    
    def getEmbedding(self, text, embed_image = False):
        if embed_image:
            print("Warning: PubMedBERT is unable to process images. Processing captions instead...")

        words = text.split()
        embeddings = [self.model[word] for word in words if word in self.model]
        if len(embeddings) > 0:
            return sum(embeddings) / len(embeddings)
        else:
            return None

class PubMedBERT_Embedder(Model):
    def __init__(self, captions_path, images_path, embeddings_json_path):
        bert_model = SentenceTransformer("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")

        super().__init__(Model_Name.PUBMED_BERT, bert_model, captions_path, images_path, embeddings_json_path)

    def preprocess_embeddings(self):
        return self.getAllEmbeddings(self.captions)
    
    def getAllEmbeddings(self, data_list, embed_images):
        if embed_images:
            print("Warning: PubMedBERT is unable to process images. Processing captions instead...")
        
        return self.model.encode(data_list)
    
    def getEmbedding(self, text, embed_image = False):
        embedding = self.getAllEmbeddings([text], False)
        return embedding


