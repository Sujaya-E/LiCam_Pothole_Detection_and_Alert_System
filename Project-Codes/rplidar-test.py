from rplidar import RPLidar, RPLidarException
import os

# Define the COM port and baud rate
PORT = 'COM6'  # Replace with your COM port (e.g., COM12 or /dev/ttyUSB0)
BAUDRATE = 256000  # Baud rate for Slamtec RPLIDAR A2M12
OUTPUT_FILE = 'C:/Slamtec/rplidar-vsc/lidar-calib-4.txt'  # File to save the scan data

# Angle range for capturing data
ANGLE_RANGE = range(0, 31)  # Angles from 0 to 30 degrees

def create_file_if_not_exists(file_path):
    """Creates the file if it does not exist."""
    if not os.path.exists(file_path):
        print(f"File '{file_path}' does not exist. Creating it...")
        with open(file_path, 'w') as file:
            file.write("Angle (deg), Distance (mm)\n")  # Write header

def main():
    try:
        # Ensure the output file exists before scanning
        create_file_if_not_exists(OUTPUT_FILE)

        # Initialize the RPLidar object
        print("Connecting to RPLIDAR...")
        lidar = RPLidar(PORT, baudrate=BAUDRATE)

        # Allow some time for initialization
        print("Initializing LiDAR...")

        # Dictionary to store unique angle data
        angle_data = {angle: None for angle in ANGLE_RANGE}  # Initialize with None for all angles

        print("Starting LiDAR scan for angles 0-30 degrees...")

        for scan in lidar.iter_scans():
            for quality, angle, distance in scan:
                rounded_angle = round(angle)
                if rounded_angle in ANGLE_RANGE and angle_data[rounded_angle] is None:
                    angle_data[rounded_angle] = distance

                # Break if all angles in the range are captured
                if all(value is not None for value in angle_data.values()):
                    break

            if all(value is not None for value in angle_data.values()):
                break

        # Write the captured data to the file
        with open(OUTPUT_FILE, 'w') as file:
            file.write("Angle (deg), Distance (mm)\n")
            for angle in sorted(angle_data):
                distance = angle_data[angle]
                if distance is not None:
                    file.write(f"{angle}, {distance:.2f}\n")

        print("All angles from 0-30 degrees have been recorded in the text file.")

    except RPLidarException as e:
        print(f"RPLIDAR exception: {e}")

    except Exception as e:
        print(f"General exception: {e}")

    finally:
        # Cleanup: Stop scanning and disconnect
        print("Stopping LiDAR and cleaning up...")
        try:
            lidar.stop()
            lidar.disconnect()
            print("LiDAR successfully stopped and disconnected.")
        except Exception as cleanup_error:
            print(f"Error during cleanup: {cleanup_error}")

if __name__ == '__main__':
    main()
