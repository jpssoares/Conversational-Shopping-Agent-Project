import json
import source.conversation.gpt as gpt
from source.conversation.predefined_messages import GET_ELEM_PROMPT


def word_for_position(pos):
    last_digit_unsigned = abs(pos) % 10

    if last_digit_unsigned == 1:
        return str(pos) + "st"
    elif last_digit_unsigned == 2:
        return str(pos) + "nd"
    elif last_digit_unsigned == 3:
        return str(pos) + "rd"
    else:
        return str(pos) + "th"


def build_answer_based_on_intent(elem, intent, result):

    if intent == "user_qa_product_measurement":
        # FIXME: make this information available in the database
        return "We dont have information about meausuremnts in our database."
    elif intent == "user_qa_product_composition":
        # FIXME: make this information available in the database
        return "We dont have information about composition in our database."
    elif intent == "user_qa_product_description":
        return (
            "For the "
            + word_for_position(elem)
            + " product, the description is "
            + result["description"]
        )
    # intent == 'user_qa_check_information' or 'user_qa_product_information'
    else:
        # FIXME: add more info in case its made available
        return (
            "For the "
            + word_for_position(elem)
            + " product: "
            + "the description is "
            + result["description"]
            + "; "
            + "the brand is "
            + result["brand"]
            + "."
        )


def get_qa_answer(intent, results, input_msg):

    # first get the element that the user wants
    gpt_answer = gpt.get_gpt_answer(GET_ELEM_PROMPT.format(input=input_msg)).replace(
        "'", '"'
    )
    # print(gpt_answer)
    elem_json = json.loads(gpt_answer)
    elem = elem_json["element"]

    # build response based on element and the intent
    if elem == "last":
        elem = len(results)

    if elem == "unknown":
        # ProductQAError: probably will never be called.
        return "Sorry I can't find that product. Try asking for the brand of the first product..."
    elif elem == "all":
        final_str = ""

        for idx, result in enumerate(results):
            final_str = (
                final_str + build_answer_based_on_intent(idx + 1, intent, result) + " "
            )
        return final_str
    else:
        if type(elem) == int:
            return build_answer_based_on_intent(elem, intent, results[elem - 1])

    return "ERROR MSG"
