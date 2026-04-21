import os
import ast
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lime.lime_text import LimeTextExplainer
import json
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

class SDOHMultiHeadModel(nn.Module):
    def __init__(self, base_model, num_labels_per_head):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base_model)
        self.dropout = nn.Dropout(0.1) 
        self.heads = nn.ModuleList([
            nn.Linear(self.bert.config.hidden_size, n_classes) 
            for n_classes in num_labels_per_head
        ])

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(out.pooler_output)
        logits = [head(pooled_output) for head in self.heads]
        return logits

class SDOHExplainerLIME:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        self.model = SDOHMultiHeadModel(args.base_model, [5] * 8)
        self.model.load_state_dict(torch.load(args.model_path, map_location=self.device))
        self.model.to(self.device)
        self.lime_explainer = LimeTextExplainer(class_names=[f"h{h}_c{c}" for h in range(8) for c in range(5)])
        self.label_cols = [
            'sdoh_community_present', 'sdoh_community_absent', 'sdoh_education', 
            'sdoh_economics', 'sdoh_environment', 'behavior_alcohol', 
            'behavior_tobacco', 'behavior_drug'
        ]
        self.threshold_range = [round(x * 0.01, 2) for x in range(0, 41)] # for thresholding from 0 to 0.4

    def predict_probs(self, texts):
        """LIME expects a function that takes a list of strings and returns a 2D array of probabilities."""
        all_probs = []
        self.model.eval()
        for text in texts:
            inputs = self.tokenizer(
                str(text), add_special_tokens=True, max_length=512, 
                padding='max_length', truncation=True, return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                logits_list = self.model(inputs['input_ids'], inputs['attention_mask'])
                probs_flat = []
                for head_logits in logits_list:
                    pooled = torch.max(head_logits, dim=0)[0] 
                    probs = F.softmax(pooled, dim=0).cpu().numpy()
                    probs_flat.extend(probs)
                all_probs.append(np.array(probs_flat))
        return np.array(all_probs)

    def run(self):
        df = pd.read_csv(self.args.input_csv)
        text_col = 'text_sp' if self.args.language == 'spanish' else 'text'
        id_col = 'file' if 'file' in df.columns else None
        nested_results = {}

        for idx in tqdm(range(len(df)), desc="LIME Explanations"):
            instance_key = str(df[id_col].iloc[idx]) if id_col else f"instance_{idx}"
            text = str(df[text_col].iloc[idx])
            
            probs_flat = self.predict_probs([text])[0]
            probs_matrix = probs_flat.reshape(8, 5)
            preds = np.argmax(probs_matrix, axis=1)
            
            target_labels = [h_idx * 5 + preds[h_idx] for h_idx in range(8)] # 7 heads (categories), 5 classes max.
            
            instance_evidence = {}

            exp = self.lime_explainer.explain_instance( # explanations for a certain text and a certain category
                text, 
                self.predict_probs, 
                labels=target_labels,
                num_features=50, 
                num_samples=self.args.max_evals 
            )

            for h_idx, h_name in enumerate(self.label_cols):
                p_class = int(preds[h_idx])
                confidence = round(float(probs_matrix[h_idx][p_class]), 4)
                label_idx = h_idx * 5 + p_class
                
                all_features = exp.as_list(label=label_idx)
                
                thresh_data = {}
                for t in self.threshold_range:
                    keywords = [
                        word for word, weight in all_features
                        if weight >= t and weight > 0 # make sure it's positive and apply threshold
                    ]
                    thresh_data[f"thresh_{t}"] = keywords
                
                instance_evidence[h_name] = {
                    "prediction": p_class, 
                    "confidence": confidence,  # confidence of classification
                    "thresholds": thresh_data  # word weight thresholding
                }
            
            nested_results[instance_key] = instance_evidence

        with open(self.args.output_json, 'w', encoding='utf-8') as f:
            json.dump(nested_results, f, ensure_ascii=False, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_evals", type=int, default=300) 
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--base_model", type=str, default="IIC/RigoBERTa-Clinical")
    parser.add_argument("--output_json", type=str, default="explanations.json")
    parser.add_argument("--language", type=str, default="spanish")
    args = parser.parse_args()
    SDOHExplainerLIME(args).run()

if __name__ == "__main__":
    main()
