import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    torch.set_default_device("cpu")

    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

    inputs = tokenizer(""""
    This data table represents a line chart:
     "TITLE | Profit ",
          "  | Software | Games ",
          " 2011 | 1070 | 1574 ",
          " 2012 | 1190 | 1801 ",          " 2013 | 784 | 2358 ",
          " 2014 | 1233 | 2033 ",
          " 2015 | 1396 | 2389 ",
          " 2016 | 1561 | 3011 ",
          " 2017 | 1104 | 2284 ",
          " 2018 | 514 | 1451 ",
          " 2019 | 1213 | 3000"
    What was the average profit on software across all years?
    
    """, return_tensors="pt", return_attention_mask=False)

    outputs = model.generate(**inputs, max_length=200)
    text = tokenizer.batch_decode(outputs)[0]
    print(text)

if __name__ == "__main__":
    main()