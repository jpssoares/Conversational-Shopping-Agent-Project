import base64
import io

from opensearchpy import OpenSearch
import os
import re
from dotenv import load_dotenv
from source.Encoder import Encoder
from itertools import chain
from PIL import Image
import spacy

load_dotenv()

host = "api.novasearch.org"
port = 443
index_name = "farfetch_images"

user = os.getenv("API_USER")
password = os.getenv("API_PASSWORD")

search_used = "text_embeddings"

search_types = [
    "full_text",
    "boolean_search",
    "text_and_attrs",
    "text_embeddings",
    "image_embeddings",
    "cross_modal_embeddings",
]

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

error_search = {
    "has_response": True,
    "recommendations": None,
    "response": "Sorry, I couldn't find any products that meet your query...",
    "system_action": "",
}

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
nlp = spacy.load("en_core_web_sm")
negation_words = set(
    [
        "no",
        "without",
        "not",
        "none",
        "neither",
        "nor",
        "never",
        "nobody",
        "nothing",
        "nowhere",
    ]
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
    print(results)
    recommendations = get_recommendations(results)
    if len(recommendations) == 0 or query_denc is None:
        return error_search
    else:
        responseDict = {
            "has_response": True,
            "recommendations": recommendations,
            "response": "Here are some items I found...",
            "system_action": "",
        }

    return responseDict, recommendations


def search_products_with_text_and_attributes(qtxt, size_of_query=3):
    qtxt_array = qtxt.split(" ")

    # verify that array has even len
    if len(qtxt_array) % 2 != 0:
        return error_search

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
        "_source": product_fields,
        "query": {"query_string": {"query": result_query}},
    }

    return get_client_search(query_denc)


def search_products_full_text(qtxt: str, size_of_query=3):
    qtxt = re.sub("\s*(not|no|without)\s+", "-", qtxt)
    qtxt = re.sub("\s*or\s+", "|", qtxt)
    qtxt = re.sub("\s*and\s+|\s+", "+", qtxt)
    query_denc = {
        "size": size_of_query,
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
        "_source": product_fields,
        "query": {"bool": bool_dict},
    }

    return get_client_search(query_denc)


def negated_tokens(token):
    descriptors = set(["pobj", "compound", "acomp", "amod", "attr"])
    if token.text == "without":
        root_child = next(token.children)
        negated_tokens = [root_child.text] + [
            child.text
            for child in root_child.children
            if child != token and child.dep_ in descriptors
        ]
        return negated_tokens
    elif token.text == "no" or token.dep_ == "neg":
        negated_tokens = [token.head.text] + [
            child.text
            for child in token.head.children
            if child != token and child.dep_ in descriptors
        ]
        return negated_tokens
    else:
        return list()


def text_embeddings_search(search_query, size_of_query=3):
    try:
        try:
            negated_terms = next(
                (
                    chain(
                        [
                            negated_tokens(tok)
                            for tok in nlp(search_query)
                            if tok.dep_ == "neg" or tok.text in negation_words
                        ]
                    )
                )
            )
        except StopIteration:
            negated_terms = []
        stop_words = [tok.text for tok in nlp(search_query) if tok.is_stop]
        undesired_terms = set(negated_terms + stop_words)
        desired_terms = set(search_query.split()) - undesired_terms - negation_words
        desired_query = " ".join(desired_terms)
        undesired_query = " ".join(undesired_terms)
    except Exception as e:
        print(
            f"Unexpected error during negation processing: {e}",
            "Defaulting to raw query.",
            sep="\n",
        )
        desired_query = search_query
        undesired_query = ""

    print(f"Desired query: '{desired_query}'", f"without '{undesired_query}'", sep="\n")
    desired_query_emb = encoder.encode(desired_query)
    undesired_query_emb = encoder.encode(undesired_query)
    desired_query_denc = {
        "size": 20 + size_of_query,
        "_source": product_fields,
        "query": {
            "knn": {
                "combined_embedding": {
                    "vector": desired_query_emb[0].detach().numpy(),
                    "k": 2,
                }
            }
        },
    }
    undesired_query_denc = {
        "size": 20,
        "_source": product_fields,
        "query": {
            "knn": {
                "combined_embedding": {
                    "vector": undesired_query_emb[0].detach().numpy(),
                    "k": 2,
                }
            }
        },
    }

    desired_items, _ = get_client_search(desired_query_denc)
    undesired_items_ids = [
        recommendation.get("id", -1)
        for recommendation in get_client_search(undesired_query_denc)[0].get(
            "recommendations", list()
        )
    ]
    desired_items["recommendations"] = [
        recommendation
        for recommendation in desired_items.get("recommendations", list())
        if recommendation.get("id", -1) not in undesired_items_ids
    ][:size_of_query]
    # print(desired_items)

    return desired_items, desired_items["recommendations"]


def decode_img(input_image_query):
    q_image = base64.b64decode(input_image_query.split(",")[1])
    image = Image.open(io.BytesIO(q_image))
    return image


def image_embeddings_search(input_image_query):
    img = decode_img(input_image_query)
    emb_img = encoder.process_image(img)

    query_denc = {
        "size": 3,
        "_source": product_fields,
        "query": {
            "knn": {"image_embedding": {"vector": emb_img[0].detach().numpy(), "k": 2}}
        },
    }

    return get_client_search(query_denc)


def cross_modal_search(input_text_query, input_image_query):
    try:
        try:
            negated_terms = next(
                (
                    chain(
                        [
                            negated_tokens(tok)
                            for tok in nlp(input_text_query)
                            if tok.dep_ == "neg" or tok.text in negation_words
                        ]
                    )
                )
            )
        except StopIteration:
            negated_terms = []
        stop_words = [tok.text for tok in nlp(input_text_query) if tok.is_stop]
        undesired_terms = set(negated_terms + stop_words)
        desired_terms = set(input_text_query.split()) - undesired_terms - negation_words
        desired_query = " ".join(desired_terms)
        undesired_query = " ".join(undesired_terms)
    except Exception as e:
        print(
            f"Unexpected error during negation processing: {e}",
            "Defaulting to raw query.",
            sep="\n",
        )
        desired_query = input_text_query
        undesired_query = ""

    print(f"Desired query: '{desired_query}'", f"without '{undesired_query}'", sep="\n")
    image = decode_img(input_image_query)

    cross_modal_embs_desired = encoder.encode_cross_modal(desired_query, image)
    cross_modal_embs_undesired = encoder.encode_cross_modal(undesired_query, image)

    desired_query_denc = {
        "size": 20 + 3,
        "_source": product_fields,
        "query": {
            "knn": {
                "combined_embedding": {
                    "vector": cross_modal_embs_desired,
                    "k": 2,
                }
            }
        },
    }
    undesired_query_denc = {
        "size": 20,
        "_source": product_fields,
        "query": {
            "knn": {
                "combined_embedding": {"vector": cross_modal_embs_undesired, "k": 2}
            }
        },
    }

    desired_items = get_client_search(desired_query_denc)
    undesired_items_ids = [
        recommendation.get("id", -1)
        for recommendation in get_client_search(undesired_query_denc).get(
            "recommendations", list()
        )
    ]
    desired_items["recommendations"] = [
        recommendation
        for recommendation in desired_items.get("recommendations", list())
        if recommendation.get("id", -1) not in undesired_items_ids
    ][:3]
    print(desired_items)

    return desired_items, desired_items["recommendations"]


def create_query_from_key_value_pais(keys, values):
    result_query = ""
    idx = 0
    while idx < len(keys):
        result_query = result_query + keys[idx] + " " + values[idx]
        idx = idx + 1
        if idx < len(keys):
            result_query = result_query + " "

    return result_query

def create_response_for_query(input_text_query, input_image_query, keys, values):
    # query_from_values = " ".join(values) # can use this one instead, but it has the same accuracy
    query_from_key_value_pairs = create_query_from_key_value_pais(keys, values)
    
    if search_used == "full_text":
        return search_products_full_text(query_from_key_value_pairs)
    elif search_used == "boolean_search":
        return search_products_boolean(query_from_key_value_pairs)
    elif search_used == "text_and_attrs":
        return search_products_with_text_and_attributes(input_text_query)
    else:
        if input_image_query == "" or input_image_query is None:
            return text_embeddings_search(query_from_key_value_pairs)
        else:
            if input_text_query == "":
                return image_embeddings_search(input_image_query)
            else:
                return cross_modal_search(query_from_key_value_pairs, input_image_query)
