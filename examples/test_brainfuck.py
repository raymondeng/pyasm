# -*- coding: utf-8 -*-

"""
    call
    ~~~~

    :copyright: 2008 by Florian Boesch <pyalot@gmail.com>.
    :license: GNU AGPL v3 or later, see LICENSE for more details.
"""

import re
import sys
from ctypes import c_ubyte
from cStringIO import StringIO

from pyasm import Program, Label
from pyasm.data import function
from pyasm.instructions import push, mov, ret, pop, call, dec, inc, sub, add, jbe, cmp, je, jne, jmp, cmova, cmovb, jae
from pyasm.registers import al, eax, ebx, ecx, edx, esp, ebp

def strip(source):
    result = ''
    for c in source:
        if c in '+-<>[],.':
            result += c
    return result

def loops(source):
    stack = list()
    for i, c in enumerate(source):
        if c == '[':
            stack.append(i)
        elif c == ']':
            start = stack.pop(-1)
            yield start, i

def inner_loops(source):
    for start, end in loops(source):
        loop = source[start+1:end]
        if '[' not in loop and ']' not in loop:
            yield start, end

def balanced_loop(source):
    for start, end in inner_loops(source):
        loop = source[start+1:end]
        if loop.count('<') == loop.count('>'):
            yield start, end, loop

def optimize_loops(source):
    result = list()
    progress = 0
    for start, end, fragment in balanced_loop(source):
        if fragment == '-':
            if source[progress:start]:
                result.append(source[progress:start])
            result.append(set_zero)
            progress = end + 1
        elif '.' not in fragment and ',' not in fragment:
            if source[progress:start]:
                result.append(source[progress:start])
            result.append(Calculation(fragment))
            progress = end + 1
    if source[progress:]:
        result.append(source[progress:])
    return result

def optimize_repetition(source):
    result = list()
    matcher = re.compile(r'\+\++|--+|<<+|>>+')
    for item in source:
        if isinstance(item, str):
            progress = 0
            for repetition in matcher.finditer(item):
                start, end = repetition.start(), repetition.end()
                repetition = item[start:end]
                if item[progress:start]:
                    result.append(item[progress:start])
                result.append(Repetition(repetition))
                progress = end
            if item[progress:]:
                result.append(item[progress:])
        else:
            result.append(item)
    return result

class SetZero:
    def __call__(self, prog):
        prog.add(
            mov(ebp.addr, 0, width=8),
        )
    def __repr__(self):
        return '0'

set_zero = SetZero()

class Repetition:
    def __init__(self, fragment):
        self.fragment = fragment
        if fragment[0] == '+':
            self.size = len(fragment) % 256
            self.action = self.increment
        elif fragment[0] == '-':
            self.size = len(fragment) % 256
            self.action = self.decrement
        elif fragment[0] == '<':
            self.size = len(fragment) % 30000
            self.action = self.dec_pointer
        elif fragment[0] == '>':
            self.size = len(fragment) % 30000
            self.action = self.inc_pointer

    def __call__(self, prog):
        self.action(prog)
    
    def increment(self, prog):
        prog.add(
            add(ebp.addr, self.size, width=8),
        )

    def decrement(self, prog):
        prog.add(
            sub(ebp.addr, self.size, width=8),
        )

    def inc_pointer(self, prog):
        label = Label()
        prog.add(
            add(ebp, self.size),
            cmp(ebp, ebx),
            jbe(label),
            sub(ebp, 30000),
            label,
        )

    def dec_pointer(self, prog):
        label = Label()
        prog.add(
            sub(ebp, self.size),
            cmp(ebp, ecx),
            jae(label),
            add(ebp, 30000),
            label,
        )

    def __repr__(self):
        return '%s%i' % (self.fragment[0], self.size)

class Calculation:
    re = re.compile(r'\++|-+|<|>')
    def __init__(self, fragment):
        offset = 0
        cells = dict()
        for item in self.re.findall(fragment):
            if item == '>':
                offset += 1
            elif item == '<':
                offset -= 1
            else:
                cells[offset] = item

        cells.pop(0)

        self.cells = dict()
        for cell, action in cells.items():
            factor = len(action)
            if factor in self.cells:
                self.cells[factor].append((action[0], cell))
            else:
                self.cells[factor] = [(action[0], cell)]

    def __call__(self, prog):
        for factor, cells in self.cells.items():
            prog.add(mov(al, ebp.addr))
            for _ in range(factor-1):
                prog.add(
                    add(al, ebp.addr),
                )
            for action, cell in cells:
                label = Label()
                if action == '+':
                    action = add(edx.addr, al)
                elif action == '-':
                    action = sub(edx.addr, al)

                if cell > 0:
                    prog.add(
                        mov(edx, ebp),
                        add(edx, cell),
                        cmp(edx, ebx),
                        jbe(label),
                        sub(edx, 30000),
                        label,
                        action,
                    )
                elif cell < 0:
                    prog.add(
                        mov(edx, ebp),
                        sub(edx, abs(cell)),
                        cmp(edx, ecx),
                        jae(label),
                        add(edx, 30000),
                        label,
                        action,
                    )

        prog.add(mov(ebp.addr, 0, width=8))

    def __repr__(self):
        return 'T'

class Brainfuck(object):
    def __init__(self, stdin=sys.stdin, stdout=sys.stdout):
        self.stdin = stdin
        self.stdout = stdout

        self.get_byte = function(c_ubyte)(self.get_byte)
        self.put_byte = function(None, c_ubyte)(self.put_byte)
        self.symbol = {
            '+' : self.increment,
            '-' : self.decrement,
            '>' : self.inc_pointer,
            '<' : self.dec_pointer,
            '[' : self.loop_start,
            ']' : self.loop_end,
            '.' : self.output,
            ',' : self.input,
        }

    def get_byte(self):
        return ord(self.stdin.read(1))

    def put_byte(self, byte):
        self.stdout.write(chr(byte))
        #self.stdout.write(str(byte)+'\n')

    def increment(self, prog, jumps):
        prog.add(
            inc(ebp.addr, width=8),
        )

    def decrement(self, prog, jumps):
        prog.add(
            dec(ebp.addr, width=8),
        )

    def inc_pointer(self, prog, jumps):
        prog.add(
            inc(ebp),
            cmp(ebp, ebx),
            cmova(ebp, ecx),
        )

    def dec_pointer(self, prog, jumps):
        prog.add(
            dec(ebp),
            cmp(ebp, ecx),
            cmovb(ebp, ebx),
        )

    def loop_start(self, prog, jumps):
        start = Label()
        end = Label()
        prog.add(
            cmp(ebp.addr, 0, width=8),
            je(end),
            start,
        )
        jumps.append((start, end))

    def loop_end(self, prog, jumps):
        start, end = jumps.pop(-1)
        prog.add(
            cmp(ebp.addr, 0, width=8),
            jne(start),
            end,
        )

    def output(self, prog, jumps):
        prog.add(
            push(ebx),
            push(ecx),
            mov(eax, 0),
            mov(al, ebp.addr),
            push(eax),
            mov(eax, self.put_byte),
            call(eax),
            add(esp, 4),
            pop(ecx),
            pop(ebx),
        )

    def input(self, prog, jumps):
        prog.add(
            push(ebx),
            push(ecx),
            mov(eax, self.get_byte),
            call(eax),
            mov(ebp.addr, al),
            pop(ecx),
            pop(ebx),
        )

    def assemble(self, source):
        if not isinstance(source, str):
            source = source.read()

        source = strip(source)
        source = optimize_loops(source)
        source = optimize_repetition(source)

        jumps = list()
        prog = Program()
        label = Label()
        prog.add(
            push(ebp),
            sub(esp, 30000),
            mov(ebp, esp),
            mov(ecx, ebp),
            mov(ebx, ebp),
            add(ebx, 30000-1),
            mov(eax, ebp),
            label,
            mov(eax.addr, 0, width=8),
            inc(eax),
            cmp(eax, ebx),
            jbe(label),
        )
       
        for item in source:
            if isinstance(item, str):
                for c in item:
                    action = self.symbol.get(c)
                    if action:
                        action(prog, jumps)
            else:
                item(prog)

        prog.add(
            add(esp, 30000),
            pop(ebp),
            ret(),
        )
        return prog

def test_hello():
    output = StringIO()
    bf = Brainfuck(stdout=output)
    prog = bf.assemble('++++++++++[>+++++++>++++++++++>+++>+<<<<-]>++.>+.+++++++..+++.>++.<<+++++++++++++++.>.+++.------.--------.>+.>.')
    fun = prog.compile()
    fun()
    assert output.getvalue() == 'Hello World!\n'

def test_mandelbrot():
    output = StringIO()
    bf = Brainfuck(stdout=output)
    prog = bf.assemble(open('mandelbrot.bf'))
    fun = prog.compile()
    fun()
    assert output.getvalue().strip() == open('mandelbrot.result').read().strip()

if __name__ == '__main__':
    import time
    bf = Brainfuck()
    #prog = bf.assemble('+++[->+>++<<].>.>.')
    #prog = bf.assemble('++++++++++[>+++++++>++++++++++>+++>+<<<<-]>++.>+.+++++++..+++.>++.<<+++++++++++++++.>.+++.------.--------.>+.>.')
    prog = bf.assemble(open('mandelbrot.bf'))
    #prog = bf.assemble(open('test.bf'))
    #prog = bf.assemble('+++[-.]')
    fun = prog.compile()
    #print prog
    start = time.time()
    fun()
    print time.time() - start

