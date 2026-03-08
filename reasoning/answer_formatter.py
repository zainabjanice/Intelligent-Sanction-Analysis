def format_aggregated_answer(question, records, group_by):
    lines = []
    for r in records:
        lines.append(f"{r[group_by]} = {r['count']}")

    return (
        f"Answer to: {question}\n\n"
        + "\n".join(lines)
    )
