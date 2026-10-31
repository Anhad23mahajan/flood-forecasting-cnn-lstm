result_rows = []

for dataset, info in DATASETS.items():
    region = info["region"]
    scale = info["scale"]

    tests = {"Test 1: rivers on the land-cover map": rivers_map(region),
             "Test 2: valleys in the ground-height map": valleys_map(region, scale)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    fig.suptitle(dataset + ": which position lines up best with the flood?", fontsize=15)

    best_moves = []

    for ax, (test_name, reference) in zip(axes, tests.items()):
        scores, shifts = shift_scores(dataset, reference)

        # Move with the best match
        best_i, best_j = np.unravel_index(np.nanargmax(scores), scores.shape)
        best_moves.append((shifts[best_i], shifts[best_j]))

        # Picture edges, so each square is centred on its move
        half_step = (shifts[1] - shifts[0]) / 2
        edges = [shifts[0] - half_step, shifts[-1] + half_step, shifts[-1] + half_step, shifts[0] - half_step]

        picture = ax.imshow(scores, cmap="viridis", extent=edges)
        ax.plot(0, 0, marker="*", markersize=18, color="red", linestyle="none")
        ax.plot(shifts[best_j], shifts[best_i], marker="x", markersize=14, markeredgewidth=3, color="black", linestyle="none")

        ax.set_title(test_name)
        ax.set_xlabel("Move right (30 m squares)")
        ax.set_ylabel("Move down (30 m squares)")
        fig.colorbar(picture, ax=ax, label="Match (higher = better)")

    # One legend for both pictures, under them
    legend_items = [Line2D([], [], marker="*", markersize=14, color="red", linestyle="none", label="Position we are using"),
                    Line2D([], [], marker="x", markersize=12, markeredgewidth=3, color="black", linestyle="none", label="Best position found")]
    fig.legend(handles=legend_items, loc="outside lower center", ncol=2, fontsize=11)
    plt.show()

    # A move smaller than half a depth square makes no real difference
    rivers_move, valleys_move = best_moves
    rivers_close = max(abs(rivers_move[0]), abs(rivers_move[1])) <= scale / 2
    valleys_close = max(abs(valleys_move[0]), abs(valleys_move[1])) <= scale / 2

    # Do the two tests point to the same place?
    shift_step = max(1, scale // 4)
    tests_agree = abs(rivers_move[0] - valleys_move[0]) <= shift_step and abs(rivers_move[1] - valleys_move[1]) <= shift_step

    if rivers_close and valleys_close:
        decision = "Keep the position"
    elif tests_agree:
        decision = "Both tests say move it"
    else:
        decision = "Tests disagree, look closer"

    result_rows.append({"Dataset": dataset, "Test 1 best move (down, right)": rivers_move,
                        "Test 2 best move (down, right)": valleys_move, "Decision": decision})

display(pd.DataFrame(result_rows))
