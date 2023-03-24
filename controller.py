from opensearchpy import OpenSearch
from opensearchpy import helpers

host = 'api.novasearch.org'
port = 443

index_name = "farfetch_images"

user = 'ifetch' # Add your user name here.
password = 'S48YdnMQ' # Add your user password here. For testing only. Don't store credentials in code.

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


def create_response_for_query(qtxt, size_of_query=3):

    query_denc = {
    'size': size_of_query,
    '_source': ['product_id', 'product_family', 'product_category', 'product_sub_category', 'product_gender', 
                'product_main_colour', 'product_second_color', 'product_brand', 'product_materials', 
                'product_short_description', 'product_attributes', 'product_image_path', 
                'product_highlights', 'outfits_ids', 'outfits_products'],
    'query': {
        'multi_match': {
        'query': qtxt,
        'fields': ['product_main_colour']
        }
    }
    }

    response = client.search(
        body = query_denc,
        index = index_name
    )

    results = [r['_source'] for r in response['hits']['hits']]
    print('\nSearch results:')
    results

    #pp.pprint(results)

    recommendations = get_recommendations(results)
    
    responseDict = { "has_response": True, "recommendations":recommendations, "response":"Here are some items I found...", "system_action":""}
    return responseDict