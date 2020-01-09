# ex12_1_6
# Load resources from previous exercise
from transactionsApriori import mat2transactions, print_apriori_rules
from binaryConverter import Xb, attributeNamesB
from apyori import apriori

# Given the processed data in the previous exercise this becomes easy:
T = mat2transactions(Xb,labels=attributeNamesB)

rules = apriori(T, min_support=0.5, min_confidence=0.75)

liste = print_apriori_rules(rules)