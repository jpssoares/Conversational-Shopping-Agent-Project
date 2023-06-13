from opensearchpy import OpenSearch
import os
import re
import ast
import validators
from dotenv import load_dotenv
from source.Encoder import Encoder
from source.image_handling import load_image
from PIL import Image
from typing import List, Union

load_dotenv()

host = "api.novasearch.org"
port = 443
index_name = "farfetch_images"

user = os.getenv("API_USER")
password = os.getenv("API_PASSWORD")

SEARCH_TYPES = [
    "full_text",
    "boolean_search",
    "text_and_attrs",
    "text_embeddings",
    "image_embeddings",
    "cross_modal_embeddings",
]

PRODUCT_FIELDS = [
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

encoder = Encoder()


def get_recommendations(results: List[dict]):
    recommendations = []

    for r in results:
        recommendations.append(
            create_new_recommendation(
                r.get("product_brand"),
                r.get("product_short_description"),
                r.get("product_id"),
                r.get("product_image_path"),
                r.get("product_attributes"),
                r.get("product_materials"),
            )
        )

    return recommendations


def create_new_recommendation(
    brand="None",
    desc="None",
    id="None",
    img_path="None",
    attributes="None",
    materials="None",
):
    recommendation = {
        "brand": brand,
        "description": desc,
        "id": id,
        "attributes": _parse_attributes(attributes),
        "materials": materials,
        "image_path": img_path,
    }
    return recommendation


def get_client_search(query_denc: dict):
    response = client.search(body=query_denc, index=index_name)
    results = [r["_source"] for r in response["hits"]["hits"]]

    recommendations = get_recommendations(results)
    if len(recommendations) == 0 or query_denc is None:
        return None
    else:
        return recommendations


def search_products_with_text_and_attributes(qtxt: str, size_of_query=3):
    qtxt_array = qtxt.split(" ")

    # verify that array has even len
    if len(qtxt_array) % 2 != 0:
        return None

    result_query = ""

    for idx, value in enumerate(qtxt_array):
        if idx % 2 != 0:
            continue
        qtxt = qtxt_array[idx + 1]

        if result_query != "":
            result_query = result_query + " AND "

        result_query = result_query + value + ":" + qtxt_array[idx + 1]

    query_denc = {
        "size": size_of_query,
        "_source": PRODUCT_FIELDS,
        "query": {"query_string": {"query": result_query}},
    }

    return get_client_search(query_denc)


def search_products_full_text(qtxt: str, size_of_query=3):
    qtxt = re.sub("\s*(not|no|without)\s+", "-", qtxt)
    qtxt = re.sub("\s*or\s+", "|", qtxt)
    qtxt = re.sub("\s*and\s+|\s+", "+", qtxt)
    query_denc = {
        "size": size_of_query,
        "_source": PRODUCT_FIELDS,
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


def search_products_boolean(qtxt: str, size_of_query=3):
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
        bool_dict["filter"] = {"term": {filter_part[0]: filter_part[1]}}

    query_denc = {
        "size": size_of_query,
        "_source": PRODUCT_FIELDS,
        "query": {"bool": bool_dict},
    }

    return get_client_search(query_denc)


def text_embeddings_search(search_query: str, size_of_query=3):
    search_query_emb = encoder.encode(search_query)
    search_query_denc = {
        "size": size_of_query,
        "_source": PRODUCT_FIELDS,
        "query": {
            "knn": {
                "combined_embedding": {
                    "vector": search_query_emb[0].detach().numpy(),
                    "k": 2,
                }
            }
        },
    }

    return get_client_search(search_query_denc)


def image_embeddings_search(input_image_query: Image, size_of_query=3):
    emb_img = encoder.process_image(input_image_query)

    query_denc = {
        "size": size_of_query,
        "_source": PRODUCT_FIELDS,
        "query": {
            "knn": {"image_embedding": {"vector": emb_img[0].detach().numpy(), "k": 2}}
        },
    }

    return get_client_search(query_denc)


def cross_modal_search(
    input_text_query: str, input_image_query: Image, size_of_query=3
):
    query_emb = encoder.encode_cross_modal(input_text_query, input_image_query)

    query_denc = {
        "size": size_of_query,
        "_source": PRODUCT_FIELDS,
        "query": {
            "knn": {
                "combined_embedding": {
                    "vector": query_emb,
                    "k": 2,
                }
            }
        },
    }

    return get_client_search(query_denc)


def _parse_attributes(attributes: str) -> list[dict]:
    """
    Attributes are a list serialized into a string or None.
    """
    try:
        attrs: list = ast.literal_eval(attributes)
    except ValueError:
        attrs = list()
    return attrs


def get_similar(recommendation: dict[str, Union[str, dict]], size_of_query=3):
    qtxt = " ".join(
        (
            [
                recommendation.get("brand", ""),
                recommendation.get("description", ""),
            ]
            + [material for material in recommendation.get("materials", list())]
            + [
                attr
                for attribute in _parse_attributes(
                    recommendation.get("attributes", list())
                )
                for attr in attribute.get("attribute_values", list())
            ]
        )
    )
    image_url = recommendation.get("image_path")
    image = load_image(image_url)
    results = (
        cross_modal_search(qtxt, image, size_of_query=size_of_query + 1)
        if validators.url(image_url)
        else text_embeddings_search(qtxt, size_of_query=size_of_query + 1)
    )
    results_without_the_same_item = [
        item for item in results if item.get("id", 0) != recommendation.get("id", 1)
    ]
    return results_without_the_same_item[:size_of_query]


def create_query_from_key_value_pairs(keys: List[str], values: List[str]):
    result_query = ""
    idx = 0
    while idx < len(keys):
        result_query = result_query + keys[idx] + " " + values[idx]
        idx = idx + 1
        if idx < len(keys):
            result_query = result_query + " "
    return result_query


def create_response_for_query(
    input_text_query: str,
    input_image_query: Image,
    keys: List[str],
    values: List[str],
    search_type="text_search",
):
    print(f"Creating response for query: '{input_text_query}' {input_image_query}")
    # query_from_values = " ".join(values) # can use this one instead, but it has the same accuracy
    query_from_key_value_pairs = create_query_from_key_value_pairs(keys, values)

    if search_type == "vqa_search":
        return cross_modal_search(input_text_query, input_image_query)
    elif search_type == "full_text":
        return search_products_full_text(query_from_key_value_pairs)
    elif search_type == "boolean_search":
        return search_products_boolean(query_from_key_value_pairs)
    elif search_type == "text_and_attrs":
        return search_products_with_text_and_attributes(input_text_query)
    else:
        if input_image_query == "" or input_image_query is None:
            return text_embeddings_search(query_from_key_value_pairs)
        else:
            if input_text_query == "":
                return image_embeddings_search(input_image_query)
            else:
                return cross_modal_search(query_from_key_value_pairs, input_image_query)
