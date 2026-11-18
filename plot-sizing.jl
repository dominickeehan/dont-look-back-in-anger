using Plots, Measures

# Sizes figures in points for latex.

# Figure sizes are written in points as (left + plot boxes + right, top + plot box + bottom). Every
# plot box is 160 points tall. A single plot box is 240 points wide (a 3:2 aspect ratio), and
# side-by-side plot boxes are 16 points apart and fill the 452.97-point text column. The sqrt(3)
# conversion in gr(size = (...) .* sqrt(3)) means that one point of the size is one point in latex,
# which in Plots' own lengths is
latex_point = sqrt(3) * Plots.px

# Around a plot box, GR puts the y-axis title 0.03 further left than the y tick labels, the x-axis
# title 0.015 further down than the x tick labels, and a plot title at the top of the figure, as
# fractions of the larger figure dimension (see Plots' GR backend). A plot title gets the same 0.015
# gap above the plot box.
gr_y_title_offset = 0.03
gr_x_title_offset = 0.015
gr_title_gap = 0.015
# Every title and tick label also gets 2 points of clear space from the edge of the figure. GR's
# text boxes for labels with maths, such as "Optimal weight, $w_t$", can fall 0.17 points short of
# the glyphs, so a smaller padding can leave them touching the edge.
padding = 2


# Moves the margins so that the plot boxes sit exactly left, right, top, and bottom points in from
# the edges of the figure, and gap points apart. Then checks that these spaces exactly fit the
# titles and tick labels plus their padding, so that nothing touches the edge and no space is wasted.
function place_plot_boxes!(plt; left, gap = 0, right, top, bottom)

        subplots = plt.subplots
        width, height = plt[:size] .* Plots.px
        points(length) = length.value / latex_point.value
        plot_boxes() = (Plots.prepare_output(plt); [Plots.plotarea(sp) for sp in subplots])

        # The layout is linear in the margins, so one correction places the plot boxes exactly.
        boxes = plot_boxes()
        subplots[1][:left_margin] += left * latex_point - Plots.left(boxes[1])
        for i in 2:length(subplots)
                subplots[i][:left_margin] += gap * latex_point - (Plots.left(boxes[i]) - Plots.right(boxes[i-1]))

        end
        subplots[end][:right_margin] += right * latex_point - (width - Plots.right(boxes[end]))
        for sp in subplots
                sp[:top_margin] += top * latex_point - Plots.top(boxes[1])
                sp[:bottom_margin] += bottom * latex_point - (height - Plots.bottom(boxes[1]))

        end

        boxes = plot_boxes()
        @assert isapprox(points(Plots.left(boxes[1])), left, atol = 1e-6)
        @assert isapprox(points(width - Plots.right(boxes[end])), right, atol = 1e-6)
        @assert isapprox(points(Plots.top(boxes[1])), top, atol = 1e-6)
        @assert isapprox(points(height - Plots.bottom(boxes[1])), bottom, atol = 1e-6)

        # The text sizes in points, from GR's text boxes. Every figure here is wider than it is tall.
        sp = subplots[1]
        W = maximum(plt[:size]) / sqrt(3)
        y_text = Plots.gr_axis_width(sp, sp[:yaxis]) * W # Tick labels and title.
        x_text = Plots.gr_axis_height(sp, sp[:xaxis]) * W
        Plots.GR.savestate()
        if sp[:title] == ""
                Plots.gr_set_tickfont(sp, sp[:yaxis])
                ticks = Plots.get_ticks(sp, sp[:yaxis], update = false)
                top_text = last(Plots.gr_get_ticks_size(ticks, sp[:yaxis][:rotation])) / 2 * W # How far the top tick label sticks out.
        else
                Plots.gr_set_font(Plots.titlefont(sp), sp)
                top_text = last(Plots.gr_text_size(sp[:title])) * W
        end
        Plots.GR.restorestate()

        exact(W) = (left = gr_y_title_offset * W + y_text + padding,
                    top = (sp[:title] == "" ? 0 : gr_title_gap * W) + top_text + padding,
                    bottom = gr_x_title_offset * W + x_text + padding)

        needed = exact(W)
        if !(isapprox(left, needed.left, atol = 0.01) && isapprox(top, needed.top, atol = 0.01) && isapprox(bottom, needed.bottom, atol = 0.01))
                # Keeping the plot boxes and gaps as they are, with right = left, solve for left.
                rest = points(width) - left - right
                exact_left = (gr_y_title_offset * rest + y_text + padding) / (1 - 2 * gr_y_title_offset)
                needed = exact(rest + 2 * exact_left)
                error("The titles and tick labels do not fit exactly. Keeping the plot boxes and gaps as they are, set " *
                      "left = right = $(round(needed.left, digits = 2)), top = $(round(needed.top, digits = 2)), " *
                      "and bottom = $(round(needed.bottom, digits = 2)).")
        end

        println("Plot boxes ", join(("$(round(points(Plots.width(box)), digits = 2)) × $(round(points(Plots.height(box)), digits = 2))" for box in boxes), ", "),
                " points, in a figure $(round(points(width), digits = 2)) × $(round(points(height), digits = 2)) points.")

        return plt
end
