import json
import source.conversation.gpt as gpt

get_elem_prompt = "I am building a dialog state tracking machine, and my model has a slot_key named \'element\'. \'element\' represent the position of the element in a given sequence. For example, \'what is the brand of the third product?\' will give me a value for \'element\'  that is 3. If you cant find a value for \'element\', please set is as unknown. If the user is refering to more than one position, set \'element\' as all.\nWhat would be the key-value pair for this phrase:\n\'{input}\'\nPlease return the result inseide curly brackets."

# Aux function. Only use if the language of choice is ENG
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

qa_intent_keys = [
 'user_qa_check_information',
 'user_qa_product_composition',
 'user_qa_product_description',
 'user_qa_product_information',
 'user_qa_product_measurement'
 ]

def build_answer_based_on_intent(elem, intent, result):

    if intent == 'user_qa_product_measurement':
        # FIXME: make this information available in the database
        return "We dont have information about meausuremnts in our database."
    elif intent == 'user_qa_product_composition':
        # FIXME: make this information available in the database
        return "We dont have information about composition in our database."
    elif intent == 'user_qa_product_description':
        return "For the " + word_for_position(elem) + " product, the description is " + result['description']
    #intent == 'user_qa_check_information' or 'user_qa_product_information'
    else:
        # FIXME: add more info in case its made available
        return "For the " + word_for_position(elem) + " product: " \
                + "the description is " + result['description'] + "; " \
                + "the brand is " + result['brand'] + "."

def get_qa_answer(intent, results, input_msg):
    
    # first get the element that the user wants
    gpt_answer = gpt.get_gpt_answer(get_elem_prompt.format(input=input_msg)).replace("\'","\"")
    print(gpt_answer)
    elem_json = json.loads(gpt_answer)
    elem = elem_json['element']

    # build response based on element and the intent
    if elem == "last":
        elem = len(results)

    if elem == "unknown":
        # ProductQAError: probably will never be called.
        return "Sorry I can't find that product. Try asking for the brand of the first product..." 
    elif elem == "all":
        final_str = ""

        for result in results:
            final_str = final_str + build_answer_based_on_intent(intent, result)
        return final_str
    else:
        if type(elem) == int:
            return build_answer_based_on_intent(elem, intent, results[elem-1])
        
    return "ERROR MSG"
