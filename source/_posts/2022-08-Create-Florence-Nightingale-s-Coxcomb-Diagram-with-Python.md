---
title: Create Florence Nightingale's Coxcomb Diagram with Python
date: 2022-08-09 21:26:04
tags: data-visualization,python
---

![Nightingale Coxcomb](https://drive.google.com/uc?export=view&id=1SC6wFmma_vObK9ABNvUkxF8TO7AogFVJ)

This is my implementation of the famous coxcomb diagram of excess death from Crimean War [created by Florence Nightingale in 1858](https://www.florence-nightingale.co.uk/coxcomb-diagram-1858/). It highlights the portion of death that occur from preventable cause such as disease and infection instead of direct battle-inflicted wound. This diagram been used to illustrate the power of visualization to empower decision making.

I realized that using computer to resolve the proportion makes the chart emphasize the excess death even further than what Florence Nightingale presented.

Anyone who know me in real life might now that I have been wanting to ship this diagram to a production environment and strap it into an automatically updated dataset via ETL. But alas, no suitable storytelling opportunity so far.

The dataset is publicly available as part of [r-data](https://r-data.pmagunia.com/dataset/r-dataset-package-histdata-nightingale). I just have to download it as csv and import as pandas.

The actual charting is handled by plotly.

```python
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Barpolar(
    r=diagram_df["Other"],
    name='Other',
    marker_color='rgb(0,0,0)'
))
fig.add_trace(go.Barpolar(
    r=diagram_df["Wounds"],
    name='Wounds',
    marker_color='rgb(106,81,163)'
))
fig.add_trace(go.Barpolar(
    r=diagram_df["Disease"],
    name='Disease',
    marker_color='rgb(158,154,200)'
))
fig.update_traces(text=diagram_df['Date'])
fig.update_layout(
    title='Diagram of the Causes of Mortality of the Army of the East',
    font_size=16,
    legend_font_size=14,
    width=700,
    polar={
        "angularaxis": {
            "rotation": 90,
            "tickmode": "array",
            "tickvals": list(range(0, 360, 360 // 12)),
            "ticktext": diagram_df["Year-Month"].values,
        },
        "radialaxis": {
            "visible": False
        }
    }
)
fig.show()
```

Full implementation can be found in [this collab notebook](https://colab.research.google.com/drive/1Plg-4wUkmyJDNX_oNk4dFfVwwE4AemaG?usp=sharing).

Keep learning!

## Reference

https://plotly.com/python/wind-rose-charts/