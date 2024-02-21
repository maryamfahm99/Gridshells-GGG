print("red")
    # Find which data point corresponds to the point picked:
    # we have to account for the fact that each data point is
    # represented by a glyph with several points
    point_id = picker.point_id/glyph_points.shape[0]
    # If the no points have been selected, we have '-1'
    if point_id != -1:
        # Retrieve the coordinates coorresponding to that data
        # point
        x, y, z = x1[point_id], y1[point_id], z1[point_id]
        # Move the outline to the data point.
        outline.bounds = (x-0.1, x+0.1,
                        y-0.1, y+0.1,
                        z-0.1, z+0.1)