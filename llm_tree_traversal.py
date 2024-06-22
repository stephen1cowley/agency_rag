from trees import TreeNode, test_tree
from langchain_openai import ChatOpenAI
import dotenv


simple_tree = {
    "question": "Does the person go to Cambridge?",
    "yes": {
        "question": "Was the person born in 2003?",
        "yes": "The person was born in 2003",
        "no": "The person was not born in 2003",
    },
    "no": "The person does not go to Cambridge",
}


dotenv.load_dotenv()
llm = ChatOpenAI(model="gpt-4o", temperature=0)


def json_schema(options):
    """The JSON structure of the response of the LLM at each node"""
    options_string = " ".join(options)

    return {
        "title": "decision",
        "description": "Answer to the question following evidence of the case.",
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "description": f"The option out of the possible options: {options_string}."
            },
            "reasoning": {
                "type": "string",
                "description": "The reasoning as to how the decision was made.",
            },
        },
        "required": ["decision", "reasoning"],
    }


def traverse_tree(evidence, the_tree):
    """Traverse the entire tree making calls to LLM."""
    while the_tree.conclusion is None:
        options = [answer for answer in the_tree.children]
        structured_llm = llm.with_structured_output(json_schema(options))

        cur_prompt = f"""You are an expert in England and Wales law.
        You are given the following case:

        {evidence}

        And must answer the following question:

        {the_tree.question}
        """

        response = structured_llm.invoke(cur_prompt)

        print(f"Question: {the_tree.question}")
        print(f"LLM response: {response['decision']}")
        print(f"LLM reasoning: {response['reasoning']}\n")

        the_tree = the_tree.children[response['decision']]

    print(f"Final conclusion: {the_tree.conclusion}\n")

the_tree = TreeNode(simple_tree)
evidence = "Stephen Cowley goes to Cambridge University and is 21 years old. The current year is 2024."
traverse_tree(evidence, the_tree)


the_tree = TreeNode(test_tree)
# https://en.wikipedia.org/wiki/Hill_v_Tupper
evidence = ("The Basingstoke Canal Co gave Hill an exclusive contractual licence in his lease "
"of Aldershot Wharf, Cottage and Boathouse to hire boats out. Hill did so regularly."
" Mr Tupper also occasionally allowed customers to use his boats by his Aldershot Inn"
" to bathe or fish in the canal. Hill wished to stop Tupper from doing so. He sued "
"Tupper, arguing that his lease gave him an exclusive easement and so a direct right"
" to enforce it against third parties (rather than mere licence)."
)
traverse_tree(evidence, the_tree)

# https://en.wikipedia.org/wiki/Re_Ellenborough_Park
evidence = ("Ellenborough Park is a 7.5-acre (3.0 ha) park in Weston-super-Mare (split by a minor road, not considered by either side, nor the courts consequential).[n 1] The larger park was owned in 1855 by two tenants in common who sold off outlying parts for the building of houses, and granted rights in the purchase/sale deeds to the house owners (and expressly to their successors in title) to enjoy the parkland which remained.[1]"
"The land was enjoyed freely until 1955, when Judge Danckwerts delivered his decision on a complex dispute at first instance. The knub of the case appealed centred on a monetary question affecting the land for the first time. It centred on the fact that the War Office had used the land during World War II, and compensation was due to be paid to the neighbours (if correctly alleging a proprietary interest to use the land, namely an easement) or the landowner, the trustees of the original owner if they were the sole person(s) with an owning interest (under the Compensation Defence Act 1939, section 2 (1)).[n 2]"
"The landowner (of the park), the beneficiaries of the trust of the original owners of the land, challenged the assertion of an 'easement' from the immediate neighbours enjoying the expressed right to use the park in their deeds (title), which they in practice also regularly enjoyed. They stated these neighbouring owner-occupiers (and their tenants) had only a personal advantage (a licence, with no proprietary rights), and not an easement proper (which would include proprietary rights)"
)
traverse_tree(evidence, the_tree)
