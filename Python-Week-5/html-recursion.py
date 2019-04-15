webpage = ["html", [
    ["head", [
        ["title", "Learning the Buddha"]
    ]],
    ["body", [
        ["h1", "The Four Noble Truths"],
        ["ul", [
            ["li", "What's suffering?"],
            ["li", "The cause of suffering"],
            ["li", "The end of suffering"],
            ["li", "A path to the end of suffering"]
        ]]
    ]]
]]
x = 0
def printHtml(code):
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

                #print(str(x))


#printHtml(webpage)
