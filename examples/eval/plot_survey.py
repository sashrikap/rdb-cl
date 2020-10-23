import numpy as onp
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os


medianprops = dict(linestyle="-", linewidth=2.5, color="firebrick")
colors = {
    "random": "gray",
    "infogain": "darkorange",
    "difficult": "cornflowerblue",
    "ratiomean": "peru",
    "ratiomin": "darkred",
}
matplotlib.rcParams["text.usetex"] = True
matplotlib.rc("font", family="serif", serif=["Palatino"])
sns.set(font="serif", font_scale=2)
sns.set_style(
    "white",
    {
        "font.family": "serif",
        "font.weight": "normal",
        "font.serif": ["Times", "Palatino", "serif"],
        "axes.facecolor": "white",
        "lines.markeredgewidth": 1,
    },
)


def load_scores(exp_names):
    method_scores = {}
    exp_scores = {}
    for exp_name in exp_names:
        for file in os.listdir(os.path.join(exp_dir, exp_name, "yaml")):
            if "feedbacks" in file:
                filepath = os.path.join(exp_dir, exp_name, "yaml", file)
                data = None
                print(filepath)
                exp_scores[exp_name] = {}
                with open(filepath, "r") as f:
                    data = yaml.load(f)["feedbacks"]
                for task_data in data:
                    for method, questions in task_data.items():
                        if method not in method_scores.keys():
                            method_scores[method] = []
                        if method not in exp_scores[exp_name].keys():
                            exp_scores[exp_name][method] = []
                        sc = questions[0]["score"] - questions[1]["score"]
                        method_scores[method].append(sc)
                        exp_scores[exp_name][method].append(sc)
    return method_scores, exp_scores


def plot_bar_survey(method_scores, save_name):
    all_scores = []
    for method, scores in method_scores.items():
        print(f"{method}: {onp.mean(scores)}, {scores}")
        all_scores.append(-1 * onp.array(scores))

    _, ax = plt.subplots(figsize=(10, 10))
    ax.set_ylabel("Evaluate edge-caseness on proposed task (higher is better)")
    ax.set_xlabel("Method")
    rects = ax.boxplot(
        all_scores,
        # whis=(5, 95),
        showfliers=False,
        patch_artist=True,
        medianprops=medianprops,
    )

    for patch, method in zip(rects["boxes"], method_scores.keys()):
        patch.set_facecolor(colors[method])
        patch.set_alpha(0.7)

    # plt.axis('scaled')
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    methods = list(method_scores.keys())
    x = onp.arange(len(methods)) + 1
    ax.set_xticks(x)
    ax.set_xticklabels(methods)

    save_path = os.path.join(exp_dir, save_name, "survey.png")
    plt.savefig(save_path)
    # plt.show()


def plot_2d_survey(all_scores, save_name):
    exp_name_ = list(all_scores.keys())[0]
    all_methods = list(all_scores[exp_name_].keys())

    for mi in range(len(all_methods)):
        for mj in range(mi + 1, len(all_methods)):
            method_i = all_methods[mi]
            method_j = all_methods[mj]

            scores_i = []
            scores_j = []
            for exp_name, exp_scores in all_scores.items():
                scores_i.append(-1 * onp.mean(exp_scores[method_i]))
                scores_j.append(-1 * onp.mean(exp_scores[method_j]))
                # for si in exp_scores[method_i]:
                #     scores_i.append(-1 * si)
                # for sj in exp_scores[method_j]:
                #     scores_j.append(-1 * sj)

            _, ax = plt.subplots(figsize=(10, 10))
            ax.set_ylabel(method_j)
            ax.set_xlabel(method_i)
            rects = ax.plot(scores_i, scores_j, "r^")

            # for patch, method in zip(rects["boxes"], method_scores.keys()):
            #     patch.set_facecolor(colors[method])
            #     patch.set_alpha(0.7)

            # plt.axis('scaled')
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.plot([-6, 6], [-6, 6], "b--")

            save_path = os.path.join(
                exp_dir, save_name, f"survey_{method_i}_{method_j}.png"
            )
            plt.savefig(save_path)


if __name__ == "__main__":
    exp_dir = "data/201012"
    exp_names = ["iterative_divide_initial_1v1_icra_19"]
    # exp_names = [f"iterative_divide_initial_1v1_icra_{i:02}" for i in list(range(4, 11)) + [15, 17, 18, 19, 20]]

    method_scores, exp_scores = load_scores(exp_names)

    plot_bar_survey(method_scores, exp_names[0])
    # plot_bar_survey(method_scores, "results")
    # plot_2d_survey(exp_scores, "results")
