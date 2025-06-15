class SortingAlgorithms:
    @staticmethod
    def merge_sort(arr, key=lambda x: x):
        """
        Merge sort implementation for sorting resumes
        Time Complexity: O(n log n)
        """
        if len(arr) <= 1:
            return arr
        
        mid = len(arr) // 2
        left = SortingAlgorithms.merge_sort(arr[:mid], key)
        right = SortingAlgorithms.merge_sort(arr[mid:], key)
        
        return SortingAlgorithms._merge(left, right, key)
    
    @staticmethod
    def _merge(left, right, key):
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            if key(left[i]) >= key(right[j]):   # Using >= for descending order (higher scores first)
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        result.extend(left[i:])
        result.extend(right[j:])
        return result
    
    @staticmethod
    def quick_sort(arr, key=lambda x: x):
        """
        Quick sort implementation for sorting resumes
        Time Complexity: Average O(n log n), Worst O(n²)
        """
        if len(arr) <= 1:
            return arr
        
        pivot = arr[len(arr) // 2]
        pivot_val = key(pivot)
        
        left = [x for x in arr if key(x) > pivot_val]  # Greater values (for descending order)
        middle = [x for x in arr if key(x) == pivot_val]
        right = [x for x in arr if key(x) < pivot_val]  # Lesser values
        
        return SortingAlgorithms.quick_sort(left, key) + middle + SortingAlgorithms.quick_sort(right, key)
    
    @staticmethod
    def heap_sort(arr, key=lambda x: x):
        """
        Heap sort implementation for sorting resumes
        Time Complexity: O(n log n)
        """
        def heapify(arr, n, i):
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2
            
            if left < n and key(arr[largest]) < key(arr[left]):
                largest = left
                
            if right < n and key(arr[largest]) < key(arr[right]):
                largest = right
                
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                heapify(arr, n, largest)
        
        n = len(arr)
        # Build max heap
        for i in range(n // 2 - 1, -1, -1):
            heapify(arr, n, i)
            
        # Extract elements one by one
        result = []
        for i in range(n - 1, -1, -1):
            arr[0], arr[i] = arr[i], arr[0]
            result.insert(0, arr[i])  # Insert at beginning for descending order
            heapify(arr, i, 0)
            
        return result

class KnapsackAlgorithm:
    @staticmethod
    def knapsack_dp(values, weights, capacity):
        """
        Dynamic Programming approach to 0/1 Knapsack Problem
        Used to select optimal resumes based on scores (values) and constraints (weights)
        Time Complexity: O(n*W) where n is number of items and W is capacity
        """
        n = len(values)
        # Initialize DP table
        dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
        
        # Build table in bottom-up manner
        for i in range(1, n + 1):
            for w in range(1, capacity + 1):
                if weights[i-1] <= w:
                    dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w])
                else:
                    dp[i][w] = dp[i-1][w]
        
        # Find selected items
        selected = []
        w = capacity
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i-1][w]:
                selected.append(i-1)
                w -= weights[i-1]
        
        return dp[n][capacity], selected
