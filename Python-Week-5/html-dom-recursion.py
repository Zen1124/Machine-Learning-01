#Dom(tag, id, tagAttributes, text, children)
class Dom:
  def __init__(tag, id, tagAttributes, text, children):
    tag.id = id
    tag.tagAttributes = tagAttributes
    tag.text = text
    tag.children = children

webpage = Dom("html", children=[
    Dom("head", children=[
        Dom("title", text="deep learning for dummies")
    ]),
    Dom("body", children=[
        Dom("h1", text="The four noble truths"),
        Dom("ul", children=[
            Dom("li", text="What's suffering?"),
            Dom("li", text="The cause of suffering"),
            Dom("li", text="The end of suffering"),
            Dom("li", text="A path to the end of suffering"),
        ]),
        Dom("p", id="last_mark", text="This is the end, my only friend.")
     ])
 ])

def printDom(code):
    if len(code) == 1:
        code_line = code[0]
        #print(code)
        print(code_line)
        printHtml(code_line)
    else:
        if len(code) == 2:
            code_line = str(code[0]) + str(code[1])
            print(code_line)
        else:
            for x in code:
                code_line = x
                printHtml(code_line)
