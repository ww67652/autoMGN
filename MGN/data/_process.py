import gmsh
import os
import numpy as np
import time

from collections import defaultdict

lc = 50

for filename in os.listdir():

    if not filename.endswith('.step'):
        continue

    gmsh.initialize()
    shape = gmsh.model.occ.importShapes(filename)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMin", lc)
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
    gmsh.model.mesh.generate(5)
    # gmsh.write('test.msh')

    _, positions, _ = gmsh.model.mesh.getNodes()
    _, _, elemNodeTags = gmsh.model.mesh.getElements(dim=2)
    # gmsh.model.mesh.clear()
    gmsh.model.remove()

    positions = np.array(positions).reshape((-1, 3))
    elemNodeTags = np.array(elemNodeTags).reshape((-1, 3)) - 1

    node2node = np.eye(len(positions), dtype=bool)
    node2node[elemNodeTags[:, 0], elemNodeTags[:, 1]] = True
    node2node[elemNodeTags[:, 0], elemNodeTags[:, 2]] = True
    node2node[elemNodeTags[:, 1], elemNodeTags[:, 2]] = True
    node2node |= node2node.T

    connections = np.array(np.where(node2node)).T


    # Create a dictionary to cache the element normals and areas for each node
    node_to_elements = defaultdict(list)
    for i, elem in enumerate(elemNodeTags):
        for node in elem:
            node_to_elements[node].append((i, elem))

    # Calculate windward coefficient for each connection
    windward_coefficients = []
    for i, connection in enumerate(connections):
        node1, node2 = connection

        # Get the elements connected to the two nodes
        elems1 = node_to_elements[node1]
        elems2 = node_to_elements[node2]

        # Find the elements common to the two nodes
        common_elems = set(i for i, _ in elems1).intersection(i for i, _ in elems2)

        # Calculate the windward coefficient for each common element
        element_coefficients = []
        for i in common_elems:
            elem = elemNodeTags[i]
            nodes = elem
            coords = positions[nodes]
            normal = np.cross(coords[1] - coords[0], coords[2] - coords[0])
            normal /= np.linalg.norm(normal)
            area = 0.5 * np.linalg.norm(normal) * np.linalg.norm(coords[1] - coords[0])
            element_coefficients.append(area * normal)

        if len(element_coefficients) > 0:
            # Take the average of the element normals weighted by the element areas
            normal = np.sum(element_coefficients, axis=0) / np.sum([np.linalg.norm(v) for v in element_coefficients])
            # Calculate the windward coefficient for the connection
            windward_coeff = np.linalg.norm(normal) * np.linalg.norm(positions[node2] - positions[node1]) / lc
            windward_coefficients.append(windward_coeff)

    # Convert the list of windward coefficients to a numpy array
    windward_coefficients = np.array(windward_coefficients)

    filename = os.path.splitext(filename)[0] + '_test'
    np.savez(os.path.join('%s.npz' % filename), connections=connections,
             positions=positions, winds=windward_coefficients)

gmsh.finalize()
