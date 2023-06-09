BEGGINING_MSG = "Hello! Welcome to Farfetch! What item are you looking for?"
GOODBYE_MSG = "Goodbye! If you need anything, I'll be here..."
RETRY_MSG = (
    "Sorry, I did not understand what you were trying to tell me... can we try again?"
)
ERROR_MSG = "Sorry can't help you with that. Please try again..."
SUCCESS_SEARCH_MSG = "Here are some items I found..."
BAD_SEARCH_MSG = "Sorry, I couldn't find any products that meet your query..."
HELP_MSG = (
    "Here are some commands you can use:\n"
    + "Change the search type: change_search_type <search_type> (full_text, boolean_search, text_and_attrs, emb_search)\n"
    + "Search for product using boolean filtering: must <field1> a ... <field2> b should <field3> c must_not <field4> d filter <field5> e\n"
    + "Search for Products with Text and Attributes\n<field> <query>\nExample: product_main_colour black\n"
    + "Searching for Products with Cross-Modal Spaces\n<query_w1> <query_w2>\nExample: black boots\n"
)

ARE_YOU_A_BOT_MSG = "Yes, I am a dialog manager created by Ricardo Pereira, João Soares and Artur Stopa, programmed to respond to queries and generate text that mimics human conversation."

WHO_ARE_YOU_MSG = "I am iFetch, a dialog manager created by Ricardo Pereira, João Soares and Artur Stopa, programmed to help you find out what you are looking for in this store."

WHO_DO_YOU_WORK_FOR_MSG = "I work for Farfetch, a luxury fashion platform that sells products from over 700 boutiques and brands from around the world."

WHO_MADE_YOU_MSG = "I was created by Ricardo Pereira, João Soares and Artur Stopa, three students from the MPDW course from UNL-NOVA."
MSG_SEARCH_TYPE_CHANGED = "The search type was successfully changed"
MSG_SEARCH_TYPE_CHANGE_FAILED = "That search type doesn't exist...\nTry another one"

GET_ELEM_PROMPT = "I am building a dialog state tracking machine, and my model has a slot_key named 'element'. 'element' represent the position of the element in a given sequence. For example, 'what is the brand of the third product?' will give me a value for 'element'  that is 3. If you cant find a value for 'element', please set is as unknown. If the user is refering to more than one position, set 'element' as all.\nWhat would be the key-value pair for this phrase:\n'{input}'\nPlease return the result inseide curly brackets."
GET_CLOTHING_ITEMS_PROMPT = "Please return a string array with the different clothes with their characteristics or design, based on this input:\n'{input}'\nPlease only include the array in your response. Use \" instead of ' in your response."


def missing_characteristics_response(missing_characteristics: list) -> str:
    return "".join(
        ["I need some more information to help You. "]
        + [
            f"What {characteristic} do You want? "
            for characteristic in missing_characteristics
        ]
    )
