# ElderWand-JetsonNano

ElderWand-JetsonNano is a machine learning project designed to run on the NVIDIA Jetson Nano platform. This project leverages several Python libraries and models to perform predictions based on trained data.

## Project Structure

```
ElderWand-JetsonNano/
├── venv/                # Virtual environment directory
├── ACLO_rf.pkl          # Pre-trained model file for ACLO
├── ACNO_rf.pkl          # Pre-trained model file for ACNO
├── ACPN_rf.pkl          # Pre-trained model file for ACPN
├── LICENSE              # License file
├── NPAC_rf.pkl          # Pre-trained model file for NPAC
├── main.py              # Main script for the project
├── predict.py           # Script for making predictions
├── train.py             # Script for training the models
```

## Getting Started

### Prerequisites

Ensure you have the following libraries installed:

- scikit-learn
- pandas
- joblib
- numpy
- pillow
- opencv-python
- Jetson.GPIO

You can install these dependencies using pip:

```sh
pip install scikit-learn pandas joblib numpy pillow opencv-python Jetson.GPIO
```

### Setting Up the Virtual Environment

1. Create a virtual environment:

    ```sh
    python3 -m venv venv
    ```

2. Activate the virtual environment:

    On Linux/macOS:
    ```sh
    source venv/bin/activate
    ```

    On Windows:
    ```sh
    venv\Scripts\activate
    ```

3. Install the required libraries:

    ```sh
    pip install -r requirements.txt
    ```

### Running the Project

#### Training the Models

To train the models, run the `train.py` script:

```sh
python train.py
```

This script will generate the `.pkl` files that are used for predictions.

#### Making Predictions

To make predictions using the pre-trained models, run the `predict.py` script:

```sh
python predict.py
```

#### Main Scripts

You can execute the main scripts to see the application in action:

```sh
python main.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project uses the following libraries:
- scikit-learn
- pandas
- joblib
- numpy
- pillow
- opencv-python
- Jetson.GPIO

We thank the developers and contributors of these libraries for their hard work.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

Thank you for using ElderWand-JetsonNano! If you have any questions or issues, please feel free to open an issue on the [GitHub repository](https://github.com/Jjateen/ElderWand-JetsonNano).
