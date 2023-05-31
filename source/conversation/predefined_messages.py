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

MSG_SEARCH_TYPE_CHANGED = "The search type was successfully changed"
MSG_SEARCH_TYPE_CHANGE_FAILED = "That search type doesn't exist...\nTry another one"
