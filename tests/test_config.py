import yaml as y

def test_bandit_arms():
    with open('config.yml') as cfile:
        b_config = y.load(cfile)['bandit']
    assert type(b_config['n_arms']) is int
    assert b_config['n_arms'] > 0

def test_arm_params():
    with open('config.yml') as cfile:
        a_config = y.load(cfile)['arm']
    assert type(a_config['mean']) is int or\
        type(a_config['mean']) is float
    
    assert type(a_config['var']) is int or\
        type(a_config['var']) is float
    
    assert a_config['var'] >= 0