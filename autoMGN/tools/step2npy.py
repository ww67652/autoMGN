import gmsh
import os
import numpy as np
import time

lc = 50

"""
将step格式的汽车模型数据提取mesh结点存为npz文件
"""
if __name__ == '__main__':
    gmsh.initialize()
    for filename in os.listdir():

        if not filename.endswith('.step'):
            continue

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

        np.savez(os.path.join('%s.npz' % os.path.splitext(filename)[0]), connections=connections,
                 positions=positions)

    gmsh.finalize()
