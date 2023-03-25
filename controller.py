from opensearchpy import OpenSearch
from opensearchpy import helpers
import os
import torch.nn.functional as F
#from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import transformers as tt

from dotenv import load_dotenv
load_dotenv()

# Program variables
host = 'api.novasearch.org'
port = 443
index_name = "farfetch_images"

user = os.getenv('API_USER')
password = os.getenv('API_PASSWORD')

product_fields = ['product_id', 'product_family', 'product_category', 'product_sub_category', 'product_gender', 
                'product_main_colour', 'product_second_color', 'product_brand', 'product_materials', 
                'product_short_description', 'product_attributes', 'product_image_path', 
                'product_highlights', 'outfits_ids', 'outfits_products']

client = OpenSearch(
    hosts = [{'host': host, 'port': port}],
    http_compress = True,
    http_auth = (user, password),
    url_prefix = 'opensearch',
    use_ssl = True,
    verify_certs = False,
    ssl_assert_hostname = False,
    ssl_show_warn = False
)

def get_recommendations(results):
    recommendations = []
    
    for r in results:
        recommendations.append(create_new_recommendation(
            r.get('product_brand'),
            r.get('product_short_description'),
            r.get('product_id'),
            r.get('product_image_path')
        ))

    return recommendations

def create_new_recommendation(brand='None', desc='None', id='None', img_path='None'):
    recommendation = {
        'brand' : brand,
        'description' : desc,
        'id' : id,
        'image_path' : img_path
    }
    return recommendation


def get_client_search(query_denc):
    response = client.search(
        body = query_denc,
        index = index_name
    )

    results = [r['_source'] for r in response['hits']['hits']]
    print('\nSearch results:')
    results
    recommendations = get_recommendations(results)
    
    responseDict = { "has_response": True, "recommendations":recommendations, "response":"Here are some items I found...", "system_action":""}
    return responseDict


def search_products_with_text_and_attributes(qtxt, search_field='product_main_colour', size_of_query=3):
    query_denc = {
    'size': size_of_query,
    '_source': product_fields,
    'query': {
        'multi_match': {
        'query': qtxt,
        'fields': [search_field]
        }
    }
    }

    return get_client_search(query_denc)


def searching_for_products_with_cross_modal_spaces(search_query, size_of_query=3):
    model = tt.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = tt.CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    inputs = tokenizer([search_query], padding=True, return_tensors="pt")
    text_features = F.normalize(model.get_text_features(**inputs))

    query_denc = {
    'size': size_of_query,
    '_source': product_fields,
    "query": {
            "knn": {
            "combined_embedding": {
                "vector": text_features[0].detach().numpy(),
                "k": 2
            }
            }
        }
    }

    return get_client_search(query_denc)


def create_response_for_query(input_query):
    input_query_parts = input_query.split(' ')
    
    if len(input_query_parts)==2:
        if input_query_parts[0] in product_fields:
            """
            Search for Products with Text and Attributes 
            <field> <query>
            Example: product_main_colour black
            """
            return search_products_with_text_and_attributes(input_query_parts[1], input_query_parts[0])
        else:
            """
            Searching for Products with Cross-Modal Spaces
            <query_w1> <query_w2>
            Example: black boots
            """
            return searching_for_products_with_cross_modal_spaces(input_query)
    return None
    