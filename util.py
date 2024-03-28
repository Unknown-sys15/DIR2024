
import test_cam2

def dummy_vision(program):
    import random
    for i in range(random.randint(1,8)):
        point = [random.randint(-90,90), random.randint(-90,90), random.randint(-90,90)]
        program.append(point)


def robtarget_to_string(input_string, pressure=False, moveL=False):

    index = input_string.find('9E+09')
    if index == -1:
        index = input_string.find('9000000000')
        if index == -1:
            index = len(input_string) + 1
    


    target = input_string[:index-1]
    target = target.replace('[', '')
    target = target.replace(']', '')
    target = target.replace(' ', '')
    target = target.replace(',', ';')

    if pressure:
        target += '1;'
    else:
        target += '0;'
    
    if moveL:
        target += '1;'
    else:
        target += '0;'
    
    if len(target) > 80:
        print("Target too long")
        return
    target += '0'*(80 - len(target))
    return bytes(target, 'utf-8')






if __name__ == '__main__':

    target = '[[191.48, -192.35, 46.12], [0.364208, -0.354181, -0.847813, -0.152057], [-1, 0, 0, 0], [9000000000, 9000000000, 9000000000, 9000000000, 9000000000, 9000000000]];'
    

    print(get_string_from_instruction([target, 0, 0]))