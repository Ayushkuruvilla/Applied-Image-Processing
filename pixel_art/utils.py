import networkx as nx
from PIL import Image
import itertools
import os
import pygame
import matplotlib.pyplot as plt
import math
import svgwrite
import cairosvg

HEURISTIC_RANK = 'heuristic_rank'
YUV_COLOR = 'yuv_color'
RGB_COLOR = 'rgb_color'
VORONOI_VERTICES = 'voronoi_vertices'
PART_OF = 'part_of'
IS_INTERSECTION = 'is_intersection'

def curve_length(similarity_graph, source):
    queue = []
    explored = set([])
    queue.append(source)
    while (len(queue) > 0):
        node = queue.pop(0)
        if nx.degree(similarity_graph, node) == 2:
            for neighbour in similarity_graph[node]:
                if neighbour not in explored and neighbour not in queue:
                    queue.append(neighbour)
        explored.add(node)
    score = max(len(explored) - 1, 2)
    return score

def curve_heuristic(similarity_graph, i, j):
    node_1 = (i, j)
    node_2 = (i + 1, j + 1)
    length = curve_length(similarity_graph, node_1)
    similarity_graph[node_1][node_2][HEURISTIC_RANK] = similarity_graph[node_1][node_2][HEURISTIC_RANK] + length
    node_1 = (i + 1, j)
    node_2 = (i, j + 1)
    length = curve_length(similarity_graph, node_1)
    similarity_graph[node_1][node_2][HEURISTIC_RANK] = similarity_graph[node_1][node_2][HEURISTIC_RANK] + length

def sparsity_heuristic(similarity_graph, i, j):
    cc_1 = nx.node_connected_component(similarity_graph, (i, j))
    cc_2 = nx.node_connected_component(similarity_graph, (i + 1, j))
    score = min(abs(len(cc_1) - len(cc_2)), 64)
    if len(cc_1) < len(cc_2):
        similarity_graph[(i, j)][(i + 1, j + 1)][HEURISTIC_RANK] = similarity_graph[(i, j)][(i + 1, j + 1)][HEURISTIC_RANK] + score
    elif len(cc_1) > len(cc_2):
        similarity_graph[(i + 1, j)][(i, j + 1)][HEURISTIC_RANK] = similarity_graph[(i + 1, j)][(i, j + 1)][HEURISTIC_RANK] + score

def island_heuristic(similarity_graph, i, j):
    score = 5
    if nx.degree(similarity_graph, (i, j)) == 1 or nx.degree(similarity_graph, (i + 1, j + 1)) == 1:
        similarity_graph[(i, j)][(i + 1, j + 1)][HEURISTIC_RANK] = similarity_graph[(i, j)][(i + 1, j + 1)][HEURISTIC_RANK] + score
    if nx.degree(similarity_graph, (i + 1, j)) == 1 or nx.degree(similarity_graph, (i, j + 1)) == 1:
        similarity_graph[(i + 1, j)][(i, j + 1)][HEURISTIC_RANK] = similarity_graph[(i + 1, j)][(i, j + 1)][HEURISTIC_RANK] + score
        

def render_as_png(similarity_graph, width, height, scale, filepath):
    scaled_width, scaled_height = width * scale, height * scale
    
    dwg = svgwrite.Drawing(
        filename=None,  
        debug=True,
        size=(f"{scaled_width}px", f"{scaled_height}px"),
        viewBox=f"0 0 {scaled_width} {scaled_height}"
    )
    
    polygons = dwg.add(dwg.g(id='polygons'))

    for node in similarity_graph.nodes:
        vertices = similarity_graph.nodes[node][VORONOI_VERTICES]
        scaled_vertices = [(vertex[0] * scale, vertex[1] * scale) for vertex in vertices]

        color = similarity_graph.nodes[node][RGB_COLOR]
        svg_color = f"rgb({color[0]},{color[1]},{color[2]})"

        polygon = polygons.add(dwg.polygon(points=scaled_vertices))
        polygon.fill(svg_color).stroke(svg_color, width=0.05)
    
    svg_string = dwg.tostring()
    temp_png_filepath = f"{filepath}_temp.png"
    png_filepath = f"{filepath}.png"

    # Convert SVG string to PNG
    cairosvg.svg2png(bytestring=svg_string, write_to=temp_png_filepath)

    with Image.open(temp_png_filepath) as img:
        if img.mode == "RGBA":
            img = img.convert("RGB")
            img.save(png_filepath)  # Save without alpha channel
        else:
            img.save(png_filepath)
            
                
def calculate_dist(p1, p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    return math.hypot(x2 - x1, y2 - y1)

def check_diff(color_1, color_2):
    y_threshold = 48
    u_threshold = 7
    v_threshold = 6
    y_diff = abs(color_1[0] - color_2[0])
    u_diff = abs(color_1[1] - color_2[1])
    v_diff = abs(color_1[2] - color_2[2])
    if (y_diff <= y_threshold) and (u_diff <= u_threshold) and (v_diff <= v_threshold):
        return False
    else:
        return True
    
def create_similarity_graph(img_rgb, img_yuv):
    #Create similarity_graph
    similarity_graph = nx.Graph()
    pixels_rgb = img_rgb.load()
    pixels_yuv = img_yuv.load()

    #Add nodes
    for i in range(img_yuv.width):
        for j in range(img_yuv.height):
            similarity_graph.add_node((i, j))
            similarity_graph.nodes[(i, j)][YUV_COLOR ] = pixels_yuv[i, j]
            similarity_graph.nodes[(i, j)][RGB_COLOR] = pixels_rgb[i, j]

    #Add edges
    for i in range(img_yuv.width):
        for j in range(img_yuv.height):
            current_node = (i, j)
            nodes_to_connect = []
            nodes_to_connect.append((i + 1, j))
            nodes_to_connect.append((i - 1, j))
            nodes_to_connect.append((i, j + 1))
            nodes_to_connect.append((i, j - 1))
            nodes_to_connect.append((i + 1, j + 1))
            nodes_to_connect.append((i + 1, j - 1))
            nodes_to_connect.append((i - 1, j + 1))
            nodes_to_connect.append((i - 1, j - 1))
            for neighbour_node in nodes_to_connect:
                if (similarity_graph.has_node(neighbour_node)):
                    color_current = similarity_graph.nodes[current_node][YUV_COLOR ]
                    color_neighbour = similarity_graph.nodes[neighbour_node][YUV_COLOR ]
                    if not check_diff(color_current, color_neighbour):
                        similarity_graph.add_edge(current_node, neighbour_node)
    return similarity_graph

def process_diagonal_edges(similarity_graph, width, height):
    for i in range(width - 1):
        for j in range(height - 1):
            nodes = []
            nodes.append((i, j))
            nodes.append((i + 1, j))
            nodes.append((i, j + 1))
            nodes.append((i + 1, j + 1))
            edges = [edge for edge in similarity_graph.edges(nodes) if (edge[0] in nodes and edge[1] in nodes)]
            if similarity_graph.has_edge((i, j), (i + 1, j + 1)) and similarity_graph.has_edge((i + 1, j), (i, j + 1)):
                if (len(edges) == 6):
                    similarity_graph.remove_edge((i, j), (i + 1, j + 1))
                    similarity_graph.remove_edge((i + 1, j), (i, j + 1))
                elif (len(edges) == 2):
                    similarity_graph[(i, j)][(i + 1, j + 1)][HEURISTIC_RANK] = 0
                    similarity_graph[(i + 1, j)][(i, j + 1)][HEURISTIC_RANK] = 0
                    curve_heuristic(similarity_graph, i, j)
                    sparsity_heuristic(similarity_graph, i, j)
                    island_heuristic(similarity_graph, i, j)
                    if similarity_graph[(i, j)][(i + 1, j + 1)][HEURISTIC_RANK] > similarity_graph[(i + 1, j)][(i, j + 1)][HEURISTIC_RANK]:
                        similarity_graph.remove_edge((i + 1, j), (i, j + 1))
                    else:
                        similarity_graph.remove_edge((i, j), (i + 1, j + 1))
                else:
                    "Error! Block has abnormal number of edges"
    return similarity_graph

def create_voronoi_cells(similarity_graph, img_yuv):
    #Create Voronoi Cells
    for x in range(img_yuv.width):
        for y in range(img_yuv.height):
            voronoi_cell_center_x = x + 0.5
            voronoi_cell_center_y = y + 0.5
            voronoi_cell_vertices = []
            # top left
            if similarity_graph.has_edge((x, y), (x - 1, y - 1)):
                voronoi_cell_vertices.append((voronoi_cell_center_x - 0.25, voronoi_cell_center_y - 0.75))
                voronoi_cell_vertices.append((voronoi_cell_center_x - 0.75, voronoi_cell_center_y - 0.25))
            elif similarity_graph.has_edge((x, y - 1), (x - 1, y)):
                voronoi_cell_vertices.append((voronoi_cell_center_x - 0.25, voronoi_cell_center_y - 0.25))
            else:
                voronoi_cell_vertices.append((voronoi_cell_center_x - 0.5, voronoi_cell_center_y - 0.5))
            # left
            voronoi_cell_vertices.append((voronoi_cell_center_x - 0.5, voronoi_cell_center_y))
            # bottom left
            if similarity_graph.has_edge((x, y), (x - 1, y + 1)):
                voronoi_cell_vertices.append((voronoi_cell_center_x - 0.75, voronoi_cell_center_y + 0.25))
                voronoi_cell_vertices.append((voronoi_cell_center_x - 0.25, voronoi_cell_center_y + 0.75))
            elif similarity_graph.has_edge((x, y + 1), (x - 1, y)):
                voronoi_cell_vertices.append((voronoi_cell_center_x - 0.25, voronoi_cell_center_y + 0.25))
            else:
                voronoi_cell_vertices.append((voronoi_cell_center_x - 0.5, voronoi_cell_center_y + 0.5))
            # bottom
            voronoi_cell_vertices.append((voronoi_cell_center_x, voronoi_cell_center_y + 0.5))
            # bottom right
            if similarity_graph.has_edge((x, y), (x + 1, y + 1)):
                voronoi_cell_vertices.append((voronoi_cell_center_x + 0.25, voronoi_cell_center_y + 0.75))
                voronoi_cell_vertices.append((voronoi_cell_center_x + 0.75, voronoi_cell_center_y + 0.25))
            elif similarity_graph.has_edge((x, y + 1), (x + 1, y)):
                voronoi_cell_vertices.append((voronoi_cell_center_x + 0.25, voronoi_cell_center_y + 0.25))
            else:
                voronoi_cell_vertices.append((voronoi_cell_center_x + 0.5, voronoi_cell_center_y + 0.5))

            # right
            voronoi_cell_vertices.append((voronoi_cell_center_x + 0.5, voronoi_cell_center_y))

            # top right
            if similarity_graph.has_edge((x, y), (x + 1, y - 1)):
                voronoi_cell_vertices.append((voronoi_cell_center_x + 0.75, voronoi_cell_center_y - 0.25))
                voronoi_cell_vertices.append((voronoi_cell_center_x + 0.25, voronoi_cell_center_y - 0.75))
            elif similarity_graph.has_edge((x, y - 1), (x + 1, y)):
                voronoi_cell_vertices.append((voronoi_cell_center_x + 0.25, voronoi_cell_center_y - 0.25))
            else:
                voronoi_cell_vertices.append((voronoi_cell_center_x + 0.5, voronoi_cell_center_y - 0.5))

            # top
            voronoi_cell_vertices.append((voronoi_cell_center_x, voronoi_cell_center_y - 0.5))

            similarity_graph.nodes[(x, y)][VORONOI_VERTICES] = voronoi_cell_vertices
    return similarity_graph

def calculate_valencies(similarity_graph, img_yuv):
    valency = {}
    for i in range(img_yuv.width):
        for j in range(img_yuv.height):
            voronoi_cell_vertices = similarity_graph.nodes[(i, j)][VORONOI_VERTICES]
            for vertex in voronoi_cell_vertices:
                if vertex in valency:
                    valency[vertex] = valency[vertex] + 1
                else:
                    valency[vertex] = 1
    return valency

def remove_valency_2_voronoi_points(similarity_graph, valency, img_yuv):
    for i in range(img_yuv.width):
        for j in range(img_yuv.height):
            voronoi_cell_vertices = similarity_graph.nodes[(i, j)][VORONOI_VERTICES]
            for vertex in voronoi_cell_vertices:
                x = vertex[0]
                y = vertex[1]
                if x != 0 and x != img_yuv.width and y != 0 and y != img_yuv.height:
                    if valency[vertex] == 2:
                        voronoi_cell_vertices.remove(vertex)
    return similarity_graph

def apply_chaikin_smoothing(similarity_graph, voronoi_graph, diagonal_length_threshold):
    for node in similarity_graph.nodes:
        P = similarity_graph.nodes[node][VORONOI_VERTICES]
        Q_R = []
        for i in range(len(P)):
            p_l = P[i]
            p_r = P[(i + 1) % len(P)]
            is_p_l_junction = voronoi_graph.nodes[p_l][IS_INTERSECTION]
            is_p_r_junction = voronoi_graph.nodes[p_r][IS_INTERSECTION]
            shouldSmooth = False
            cell_centers = voronoi_graph.edges[(p_l, p_r)][PART_OF]

            if (len(cell_centers) == 2) and (not is_p_l_junction) and (not is_p_r_junction):
                color_1 = similarity_graph.nodes[cell_centers[0]][YUV_COLOR ]
                color_2 = similarity_graph.nodes[cell_centers[1]][YUV_COLOR ]
                if check_diff(color_1, color_2):
                    shouldSmooth = True

            if shouldSmooth:
                factor_1 = 0.75
                factor_2 = 1.0 - factor_1
                diagonal_length = calculate_dist(p_l, p_r)
                if diagonal_length > diagonal_length_threshold:
                    factor_1 = 1.0 / 8.0
                    factor_2 = 1.0 - factor_1
                q_i = (factor_1 * p_l[0] + factor_2 * p_r[0], factor_1 * p_l[1] + factor_2 * p_r[1])
                r_i = (factor_2 * p_l[0] + factor_1 * p_r[0], factor_2 * p_l[1] + factor_1 * p_r[1])
                Q_R.append(q_i)
                Q_R.append(r_i)
            else:
                if p_l not in Q_R:
                    Q_R.append(p_l)
                if p_r not in Q_R:
                    Q_R.append(p_r)

        similarity_graph.nodes[node][VORONOI_VERTICES] = Q_R

def smooth_voronoi_graph(similarity_graph, num_iterations, num_different_colors_threshold, diagonal_length_threshold, width, height):
    """
    Performs iterative Voronoi similarity_graph refinement and Chaikin's smoothing.
    """

    for _ in range(num_iterations):
        # Step 1: Build Voronoi Graph
        voronoi_graph = nx.Graph()
        for i in range(width):
            for j in range(height):
                voronoi_cell_vertices = similarity_graph.nodes[(i, j)][VORONOI_VERTICES]
                for l in range(len(voronoi_cell_vertices)):
                    r = (l + 1) % len(voronoi_cell_vertices)
                    v1 = voronoi_cell_vertices[l]
                    v2 = voronoi_cell_vertices[r]
                    if voronoi_graph.has_edge(v1, v2):
                        voronoi_graph.edges[(v1, v2)][PART_OF].append((i, j))
                    else:
                        voronoi_graph.add_edge(v1, v2)
                        voronoi_graph.edges[(v1, v2)][PART_OF] = [(i, j)]

        # Step 2: Mark Junctions in Voronoi Graph
        for node in voronoi_graph.nodes:
            adjacent_cell_colors = set()
            edges = voronoi_graph.edges(node)
            for edge in edges:
                part_of = voronoi_graph.edges[edge][PART_OF]
                for cell_center in part_of:
                    cell_color = similarity_graph.nodes[cell_center][YUV_COLOR ]
                    adjacent_cell_colors.add(cell_color)
            adjacent_color_pairs = itertools.combinations(adjacent_cell_colors, 2)
            num_different_colors = sum(1 for pair in adjacent_color_pairs if check_diff(pair[0], pair[1]))

            voronoi_graph.nodes[node][IS_INTERSECTION] = num_different_colors > num_different_colors_threshold

        # Step 3: Apply Chaikin's Method
        apply_chaikin_smoothing(similarity_graph, voronoi_graph, diagonal_length_threshold)

    return similarity_graph

def vectorization():
            """Runs the vectorization process on the selected image."""
            output_png_path = "pixel_art/outputs/vectorized_output"
            selected_image = "input/pixel_art/smw_dolphin_input.png"
            print(f"Processing: {selected_image} -> {output_png_path}")
            image = Image.open(selected_image)
            img_converted = image.convert('YCbCr')

            # Step 1: Create Similarity Graph
            similarity_graph = create_similarity_graph(image, img_converted)

            # Step 2: Process Diagonal Edges
            similarity_graph = process_diagonal_edges(similarity_graph, img_converted.width, img_converted.height)

            # Step 3: Generate Voronoi Cells
            similarity_graph = create_voronoi_cells(similarity_graph, img_converted)

            # Step 4: Remove Low-Valency Voronoi Points
            valency = calculate_valencies(similarity_graph, img_converted)
            similarity_graph = remove_valency_2_voronoi_points(similarity_graph, valency,img_converted)

            # Step 5: Apply Voronoi Smoothing using Chaikin's Algorithm
            similarity_graph = smooth_voronoi_graph(
                similarity_graph,
                num_iterations=4,
                num_different_colors_threshold=3,
                diagonal_length_threshold=0.8,
                width=img_converted.width,
                height=img_converted.height
            )
            render_as_png(similarity_graph, img_converted.width, img_converted.height, 10, output_png_path)

#run this to just test the vectorization logic
#vectorization()