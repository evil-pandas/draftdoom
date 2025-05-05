#!/usr/bin/env python
# coding: utf-8

import gradio as gr
import numpy as np
import json
import xgboost as xgb

# 1) Load the trained model
model = xgb.Booster()
model.load_model("xgboost_v3dota2_model.json")

# 2) Build hero ↔ feature mappings
with open("heroes.json") as f:
    heroes_data = json.load(f)["heroes"]
id_to_name       = {h["id"]: h["localized_name"] for h in heroes_data}
feature_cols     = model.feature_names
hero_feature_cols= [c for c in feature_cols if c.startswith("hero_")]
feature_to_name  = {
    feat: id_to_name[int(feat.split("_")[1])]
    for feat in hero_feature_cols
    if int(feat.split("_")[1]) in id_to_name
}
name_to_feature  = {name: feat for feat, name in feature_to_name.items()}
hero_list        = list(name_to_feature.keys())

# 3) Define prediction function
def predict_win(h1, h2, h3, h4, h5):
    x = np.zeros(len(feature_cols), dtype=float)
    # hidden context
    x[ feature_cols.index("clusterid") ] = 150
    x[ feature_cols.index("gamemode")   ] = 2
    x[ feature_cols.index("gametype")   ] = 2
    # hero picks
    for name in (h1, h2, h3, h4, h5):
        feat = name_to_feature[name]
        x[ feature_cols.index(feat) ] = 1

    dmat     = xgb.DMatrix(x.reshape(1, -1), feature_names=feature_cols)
    win_pct  = model.predict(dmat)[0]
    loss_pct = 1 - win_pct
    verdict  = "Win 🟢" if win_pct > 0.5 else "Loss 🔴"

    return (
        f"⚔️ DraftDoom Verdict: {verdict}\n\n"
        f"💚 Win Probability: {win_pct:.1%}\n"
        f"❤️ Loss Probability: {loss_pct:.1%}"
    )

# 4) Launch Gradio interface
emojis = ["🛡️","⚔️","🧙","🧟","🐉"]
interface = gr.Interface(
    fn=predict_win,
    inputs=[gr.Dropdown(hero_list, label=f"{emojis[i]} Hero {i+1}") for i in range(5)],
    outputs=gr.Textbox(label="Your Fate"),
    title="⚔️ 🔥 DraftDoom 🔥",
    description="""
**Built for coaches, built from data.**  
_DraftDoom judges your lineup before the enemy does._  

**Pick wisely.**  
*The wrong five could doom you.*
""",
    flagging_mode="never"
)

if __name__ == "__main__":
    interface.launch(share=True)
