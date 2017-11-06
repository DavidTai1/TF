class KMP:
    def partial(self, pattern):
        """ Calculate partial match table: String -> [Int]"""
        ret = [0]

        for i in range(1, len(pattern)):
            j = ret[i - 1]
            while j > 0 and pattern[j] != pattern[i]:
                j = ret[j - 1]
            ret.append(j + 1 if pattern[j] == pattern[i] else j)
        return ret

    def search(self, T, P):
        """
        KMP search main algorithm: String -> String -> [Int]
        Return all the matching position of pattern string P in S
        """
        partial, ret, j = self.partial(P), [], 0
        print(partial)

        for i in range(len(T)):
            while j > 0 and T[i] != P[j]:
                j = partial[j - 1]
            if T[i] == P[j]: j += 1
            if j == len(P):
                ret.append(i - (j - 1))
                j = partial[j-1]

        return ret


# Dynamic Programming Python implementation of Min Cost Path
# problem
R = 4
C = 4
tc = [[0 for x in range(C)] for x in range(R)]
def minCost(cost, m, n):

	# Instead of following line, we can use int tc[m+1][n+1] or
	# dynamically allocate memoery to save space. The following
	# line is used to keep te program simple and make it working
	# on all compilers.


	tc[0][0] = cost[0][0]

	# Initialize first column of total cost(tc) array
	for i in range(1, m+1):
		tc[i][0] = tc[i-1][0] + cost[i][0]

	# Initialize first row of tc array
	for j in range(1, n+1):
		tc[0][j] = tc[0][j-1] + cost[0][j]
	# Construct rest of the tc array
	for i in range(1, m+1):
		for j in range(1, n+1):
			tc[i][j] = min(tc[i-1][j-1], tc[i-1][j], tc[i][j-1]) + cost[i][j]
	return tc[m][n]

# Driver program to test above functions
cost = [[10,0,0,1],
		[0,1,1,0],
		[0,1,1,0],
		[0,0,0,0]]
print(minCost(cost, 3, 3))

for i in tc:
    print(i)
# This code is contributed by Bhavya Jain
