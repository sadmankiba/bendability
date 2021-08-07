from typing import TypedDict


class Library(TypedDict):
    name: str
    file: str


LIBRARIES = {
    'cnl': {
        'name': 'cnl',
        'file': '41586_2020_3052_MOESM4_ESM.txt'
    },
    'rl': {
        'name': 'rl',
        'file': '41586_2020_3052_MOESM6_ESM.txt'
    },
    'tl': {
        'name': 'tl',
        'file': '41586_2020_3052_MOESM8_ESM.txt'
    },
    'chrvl': {
        'name': 'chrvl',
        'file': '41586_2020_3052_MOESM9_ESM.txt'
    },
    'libl': {
        'name': 'libl',
        'file': '41586_2020_3052_MOESM11_ESM.txt'
    }
}
