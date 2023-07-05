from part_a import test_student_code
import pickle

def main():

    return test_student_code()

if __name__ == "__main__":
    # print(main())

    results = main()
    student_id = results['Student_info']['Student_ID']
    with open(f'{student_id}.pkl', 'wb') as file:
        pickle.dump(results,file)