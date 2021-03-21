import matplotlib.pyplot as plt
import pandas as pd


def model_metrics(resultsDict):
    df = pd.DataFrame.from_dict(resultsDict)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    pallette = plt.cm.get_cmap("tab20c", len(df.columns))
    colors = [pallette(x) for x in range(len(df.columns))]
    color_dict = dict(zip(df.columns, colors))
    fig = plt.figure(figsize=(20, 15))

    # MAE plot
    fig.add_subplot(2, 2, 1)
    df.loc["mae"].sort_values().plot(
        kind="bar", colormap="Paired", color=[
            color_dict.get(
                x, "#333333") for x in df.loc["mae"].sort_values().index], )
    plt.legend()
    #plt.title("MAE Metric, lower is better")
    plt.title("MAE")
    fig.add_subplot(2, 2, 2)
    df.loc["rmse"].sort_values().plot(
        kind="bar", colormap="Paired", color=[
            color_dict.get(
                x, "#333333") for x in df.loc["rmse"].sort_values().index], )
    plt.legend()
    #plt.title("RMSE Metric, lower is better")
    plt.title("RMSE")
    fig.add_subplot(2, 2, 3)
    df.loc["mape"].sort_values().plot(
        kind="bar", colormap="Paired", color=[
            color_dict.get(
                x, "#333333") for x in df.loc["mape"].sort_values().index], )
    plt.legend()
    #plt.title("MAPE Metric, lower is better")
    plt.title("MAPE")
    fig.add_subplot(2, 2, 4)
    df.loc["r2"].sort_values(ascending=False).plot(
        kind="bar",
        colormap="Paired",
        color=[
            color_dict.get(x, "#333333")
            for x in df.loc["r2"].sort_values(ascending=False).index
        ],
    )
    plt.legend()
    #plt.title("R2 Metric, higher is better")
    plt.title("R2")
    plt.tight_layout()
    plt.savefig("metrics.png")
    plt.show()

    best_model = chooseBest(resultsDict)

    return best_model

def chooseBest(resultsDict):
    
    count = {}
    mae = []
    rmse = []
    r2 = []
    
    key_list = list(resultsDict.keys())
    count = dict.fromkeys(resultsDict.keys(), 1)
    #{AR: 1, HWES: 3} returning max val key
    
    for i in resultsDict:
        mae.append(resultsDict.get(i).get('mae'))
        rmse.append(resultsDict.get(i).get('rmse'))
        r2.append(resultsDict.get(i).get('r2'))
    
    find_min(mae, key_list, count)
    find_min(rmse, key_list, count)
    find_min(r2, key_list, count)
    
    best_model = max(count, key=count.get)
    return best_model

def find_min(model_list, key_list, count):
    min_val = model_list[0]
    index = 0

    for i in range(1, len(model_list)):
        if model_list[i] < min_val:
            min_val = model_list[i]
            index = i

    count_index = str(key_list[index])
    
    if count_index in count:
        count[count_index] += 1

