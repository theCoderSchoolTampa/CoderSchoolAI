def h_distance(start_pos, end_pos):
    return ((start_pos[0] - end_pos[0]) ** 2 + (start_pos[1] - end_pos[1]) ** 2) ** 0.5


class HeapQueue:
    def __init__(self, min_heap=True):
        self.heap = []
        self.min_heap = min_heap

    def size(self):
        return len(self.heap)

    def push(self, item):
        self.heap.append(item)

        position = len(self.heap) - 1
        parent = (position - 1) // 2

        if self.min_heap:
            while parent >= 0 and self.heap[position] < self.heap[parent]:
                self.heap[parent], self.heap[position] = (
                    self.heap[position],
                    self.heap[parent],
                )
                position, parent = parent, (parent - 1) // 2
        else:
            while parent >= 0 and self.heap[position] > self.heap[parent]:
                self.heap[parent], self.heap[position] = (
                    self.heap[position],
                    self.heap[parent],
                )
                position, parent = parent, (parent - 1) // 2

    def pop(self):
        if not self.heap:
            return None
        result = self.heap[0]
        self.heap[0] = self.heap[-1]
        self.heap.pop()
        position = 0
        left = 1
        right = 2

        if self.min_heap:
            while left < len(self.heap):
                min_child = left
                if right < len(self.heap) and self.heap[right] < self.heap[left]:
                    min_child = right

                if self.heap[position] > self.heap[min_child]:
                    self.heap[position], self.heap[min_child] = (
                        self.heap[min_child],
                        self.heap[position],
                    )
                    position = min_child
                    left = position * 2 + 1
                    right = position * 2 + 2
                else:
                    break
        else:
            while left < len(self.heap):
                max_child = left
                if right < len(self.heap) and self.heap[right] > self.heap[left]:
                    max_child = right

                if self.heap[position] < self.heap[max_child]:
                    self.heap[position], self.heap[max_child] = (
                        self.heap[max_child],
                        self.heap[position],
                    )
                    position = max_child
                    left = position * 2 + 1
                    right = position * 2 + 2
                else:
                    break
        return result

    def remove_worst(self):
        self.heap.pop(0)
