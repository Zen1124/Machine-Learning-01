webpage = ["html", [
    "head", [
        "title", "Learning the Buddha"
    ],
    "body", [
        ["h1", "The Four Noble Truths"]
        ["ul", [
            ["li", "What's suffering?"],
            ["li", "The cause of suffering"],
            ["li", "The end of suffering"],
            ["li", "A path to the end of suffering"],
         ]
        ]
    ]
    ]
    ]

def printHtml(code):
    for x in code:
        print(x)

printHtml(webpage)
