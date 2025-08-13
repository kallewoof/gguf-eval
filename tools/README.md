# Tools and utilities for gguf-eval

* utils.py contains help methods for e.g. evaluations
* graphs.py contains help methods for rendering graphs, and is used by ../plot.py
* import.py, convert.cpp, json.hpp are described below

## Dataset manipulation tools (WIP)

* convert.cpp is a utility originally written by Ikawrakow to convert from the llama.cpp .bin format into JSON. It has been extended to allow conversion to/from both types.
* import.py is a stand-alone tool for converting existing datasets into llama.cpp's perplexity tool format (WIP)
* json.hpp is https://github.com/nlohmann/json


This contains tools for creating or manipulating datasets. Normally you don't wanna be here unless you plan to adopt an existing dataset for use in gguf-eval.

You can compile the `convert.cpp` executable by doing

```bash
g++ convert.cpp -o convert
```

and then use it to convert to and from .bin and .json format. When you do
```bash
tools/convert dataset/x.bin dataset/y.json
```
you will convert the binary dataset `x.bin` to the json equivalent and name it `y.json`. The filename extension determines the format, of which .bin and .json are the only supported formats right now.
