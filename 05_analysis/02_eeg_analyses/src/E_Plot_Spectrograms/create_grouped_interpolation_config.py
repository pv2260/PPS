import re

def get_interpolation_key(event_name):
    """
    Extract D and speed key from event name.

    Example:
        Both_D7_Narrow_Slow
        -> D7_Narrow_Slow
    """

    match = re.search(
        r"(D\d+_Narrow_(?:Fast|Slow))",
        event_name
    )

    if match:
        return match.group(1)

    elif "VisualOnly" in event_name:
        return event_name

    return None



def create_grouped_interpolation_config(
    groups,
    interpolation_config
):
    """
    Create interpolation config for grouped conditions.
    Keeps the interpolation parameters of the first condition
    in each group.
    """

    grouped_config = {}

    for group_name, conditions in groups.items():

        # take first condition as representative
        first_condition = conditions[0]

        # remove modality prefix
        key = get_interpolation_key(first_condition)

        if key in interpolation_config:
            grouped_config[group_name] = interpolation_config[key]

        else:
            print(
                f"Missing interpolation config for {group_name}: {key}"
            )

    return grouped_config




