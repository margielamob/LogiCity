def check_fol_rule_syntax(rule):
    stack = []
    for char in rule:
        if char == '(':
            stack.append(char)
        elif char == ')':
            if not stack or stack[-1] != '(':
                return "Error: Unmatched closing parenthesis"
            stack.pop()

    if stack:
        return "Error: Unmatched opening parenthesis"

    return "Rule syntax appears to be correct"