from opensearchpy import OpenSearch
from opensearchpy import helpers
import os
import torch.nn.functional as F

# from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import transformers as tt
import re

from dotenv import load_dotenv

load_dotenv()

# Program variables
host = "api.novasearch.org"
port = 443
index_name = "farfetch_images"

user = os.getenv("API_USER")
password = os.getenv("API_PASSWORD")

search_used = "full_text"
search_types = ["full_text", "boolean_search", "text_and_attrs"]

product_fields = [
    "product_id",
    "product_family",
    "product_category",
    "product_sub_category",
    "product_gender",
    "product_main_colour",
    "product_second_color",
    "product_brand",
    "product_materials",
    "product_short_description",
    "product_attributes",
    "product_image_path",
    "product_highlights",
    "outfits_ids",
    "outfits_products",
]

client = OpenSearch(
    hosts=[{"host": host, "port": port}],
    http_compress=True,
    http_auth=(user, password),
    url_prefix="opensearch",
    use_ssl=True,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
)


def get_recommendations(results):
    recommendations = []

    for r in results:
        recommendations.append(
            create_new_recommendation(
                r.get("product_brand"),
                r.get("product_short_description"),
                r.get("product_id"),
                r.get("product_image_path"),
            )
        )

    return recommendations


def create_new_recommendation(brand="None", desc="None", id="None", img_path="None"):
    recommendation = {
        "brand": brand,
        "description": desc,
        "id": id,
        "image_path": img_path,
    }
    return recommendation


def get_client_search(query_denc):
    response = client.search(body=query_denc, index=index_name)

    results = [r["_source"] for r in response["hits"]["hits"]]
    print("\nSearch results:")
    recommendations = get_recommendations(results)
    if len(recommendations) == 0 or query_denc is None:
        responseDict = {
            "has_response": True,
            "recommendations": None,
            "response": "Sorry, I couldn't find any products that meet your query...",
            "system_action": "",
        }
    else:
        responseDict = {
            "has_response": True,
            "recommendations": recommendations,
            "response": "Here are some items I found...",
            "system_action": "",
        }

    return responseDict


def search_products_with_text_and_attributes(
    qtxt, search_field="product_main_colour", size_of_query=3
):
    query_denc = {
        "size": size_of_query,
        "_source": product_fields,
        "query": {"multi_match": {"query": qtxt, "fields": [search_field]}},
    }

    return get_client_search(query_denc)


def search_products_full_text(qtxt: str):
    qtxt = re.sub("\s*(not|no|without)\s*", "-", qtxt)
    qtxt = re.sub("\s*or\s*", "|", qtxt)
    qtxt = re.sub("\s*and\s*|\s+", "+", qtxt)
    query_denc = {
        "size": 5,
        "_source": product_fields,
        "query": {
            "simple_query_string": {
                "query": qtxt,
                "fields": [
                    "product_family",
                    "product_category",
                    "product_sub_category",
                    "product_gender",
                    "product_main_colour",
                    "product_second_color",
                    "product_brand",
                    "product_materials",
                    "product_short_description",
                    "product_attributes",
                    "product_image_path",
                ],
            }
        },
    }

    return get_client_search(query_denc)


def search_products_boolean(qtxt: str):
    try:
        must_part = re.search("must\s+(.+?)\s+should", qtxt).group(1).split(" ")
    except AttributeError:
        must_part = []
    try:
        should_part = re.search("should\s+(.+?)\s+must_not", qtxt).group(1).split(" ")
    except AttributeError:
        should_part = []
    try:
        must_not_part = re.search("must_not\s+(.+?)\s+filter", qtxt).group(1).split(" ")
    except AttributeError:
        must_not_part = []
    try:
        filter_part = qtxt.split(" filter ")[1].split(" ")
    except IndexError:
        filter_part = []

    bool_dict = {"must": [], "should": [], "must_not": []}

    if len(must_part) > 1:
        for i in range(0, len(must_part), 2):
            bool_dict.get("must").append({"match": {must_part[i]: must_part[i + 1]}})
    if len(should_part) > 1:
        for i in range(0, len(should_part), 2):
            bool_dict.get("should").append(
                {"match": {should_part[i]: should_part[i + 1]}}
            )
    if len(must_not_part) > 1:
        for i in range(0, len(must_not_part), 2):
            bool_dict.get("must_not").append(
                {"match": {must_not_part[i]: must_not_part[i + 1]}}
            )
    if len(filter_part) > 1:
        bool_dict.get("filter").get("term")[filter_part[0]] = filter_part[1]

    print(bool_dict)
    query_denc = {"size": 5, "_source": product_fields, "query": {"bool": bool_dict}}

    return get_client_search(query_denc)


def searching_for_products_with_cross_modal_spaces(search_query, size_of_query=3):
    model = tt.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = tt.CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    inputs = tokenizer([search_query], padding=True, return_tensors="pt")
    text_features = F.normalize(model.get_text_features(**inputs))

    query_denc = {
        "size": size_of_query,
        "_source": product_fields,
        "query": {
            "knn": {
                "combined_embedding": {
                    "vector": text_features[0].detach().numpy(),
                    "k": 2,
                }
            }
        },
    }

    return get_client_search(query_denc)


def create_response_for_query(input_query):
    input_query_parts = input_query.split(" ")
    if search_used == "full_text":
        return search_products_full_text(input_query)
    elif search_used == "boolean_search":
        return search_products_boolean(input_query)
    elif search_used == "text_and_attrs":
        return search_products_with_text_and_attributes(
            input_query_parts[0], input_query_parts[1]
        )
    else:
        return searching_for_products_with_cross_modal_spaces(input_query)

    return None
