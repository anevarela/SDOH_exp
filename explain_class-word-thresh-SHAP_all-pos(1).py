import os
import ast
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import shap
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

class SDOHExplainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        self.model = SDOHMultiHeadModel(args.base_model, [5] * 8)
        self.model.load_state_dict(torch.load(args.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.label_cols = [
            'sdoh_community_present', 'sdoh_community_absent', 'sdoh_education', 
            'sdoh_economics', 'sdoh_environment', 'behavior_alcohol', 
            'behavior_tobacco', 'behavior_drug'
        ]
        self.threshold_range = [round(x * 0.01, 2) for x in range(0, 41)]

    def get_predict_all_flat(self):
        """Returns a flat array of 40 probabilities (8 heads * 5 classes) for SHAP compatibility."""
        def predict(texts):
            all_probs = []
            for text in texts:
                inputs = self.tokenizer(
                    str(text), add_special_tokens=True, max_length=512, stride=128,
                    padding='max_length', truncation=True, return_overflowing_tokens=True,
                    return_tensors='pt'
                ).to(self.device)
                
                inputs.pop("overflow_to_sample_mapping", None)
                inputs.pop("offset_mapping", None)

                with torch.no_grad():
                    logits_list = self.model(inputs['input_ids'], inputs['attention_mask'])
                    probs_flat = []
                    for head_logits in logits_list:
                        pooled = torch.max(head_logits, dim=0)[0]
                        probs = F.softmax(pooled, dim=0).cpu().numpy()
                        probs_flat.extend(probs) 
                    all_probs.append(np.array(probs_flat))
            return np.array(all_probs)
        return predict

    def run(self):
        df = pd.read_csv(self.args.input_csv)
        text_col = 'text_sp' if self.args.language == 'spanish' else 'text'
        id_col = 'file' if 'file' in df.columns else None
        nested_results = {}
        
        masker = shap.maskers.Text(tokenizer=r"\s+") 
        predict_fn = self.get_predict_all_flat()
        explainer = shap.Explainer(predict_fn, masker=masker)
        
        for idx in tqdm(range(len(df)), desc="SHAP Highlights"):
            instance_key = str(df[id_col].iloc[idx]) if id_col else f"instance_{idx}"
            text = str(df[text_col].iloc[idx])
            instance_evidence = {}
            
            probs_flat = predict_fn([text])[0]
            probs_matrix = probs_flat.reshape(8, 5) 
            preds = np.argmax(probs_matrix, axis=1)
            
            shap_values = explainer([text], max_evals=self.args.max_evals, batch_size=self.args.batch_size)
            
            words = shap_values.data[0]
            reshaped_values = shap_values.values[0].reshape(len(words), 8, 5)
        
            for h_idx, h_name in enumerate(self.label_cols):
                p_class = int(preds[h_idx])
                confidence = round(float(probs_matrix[h_idx][p_class]), 4)

                word_vals = reshaped_values[:, h_idx, p_class]
                
                importance = sorted([
                    (words[i].strip(), float(word_vals[i])) for i in range(len(word_vals))
                    if word_vals[i] > 0 and len(words[i].strip()) > 1
                ], key=lambda x: x[1], reverse=True)
        
                thresh_data = {}
                for t in self.threshold_range:
                    keywords = []
                    seen = set()
                    for w, v in importance:
                        if v >= t:
                            if w.lower() not in seen:
                                keywords.append(w)
                                seen.add(w.lower())

                    thresh_data[f"thresh_{t}"] = keywords
        
                instance_evidence[h_name] = {
                    "prediction": p_class, 
                    "confidence": confidence, 
                    "thresholds": thresh_data
                }
            
            nested_results[instance_key] = instance_evidence

        # Save results
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
    SDOHExplainer(args).run()

if __name__ == "__main__":
    main()
