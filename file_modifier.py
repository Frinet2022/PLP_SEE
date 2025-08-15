def file_modifier():
  """Reads a file, modifies its content, and writes to a new file with error handling."""
input_filename = input("Enter the name of the file to read: ")
output_filename = input("Enter the name of the file to write: ")
try:
        with open(input_filename, "r") as infile:
            content = infile.read()
            word_count = len(content.split())
            upper_content = content.upper()
            
        with open(output_filename, "w") as outfile:
            outfile.write(upper_content)
            outfile.write("\n\n")
            outfile.write(f"Word Count: {word_count}\n")
            
        print(f"{output_filename} has been created successfully!")
except FileNotFoundError:
        print(f"Error: The file {input_filename} does not exist.")
except Exception as e:
        print(f"An error occurred: {e}")
if __name__ == "__main__":
    file_modifier()



  
  
