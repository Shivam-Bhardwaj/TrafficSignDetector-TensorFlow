# Tutorial: Building and Deploying a Traffic Sign Classifier Web App

This tutorial will guide you through the process of building a deep learning web application with Flask and deploying it to Vercel.

## 1. Project Structure

Here's an overview of the key files:

-   `app.py`: The Flask web application.
-   `lenet.keras`: The pre-trained TensorFlow model.
-   `requirements.txt`: The Python dependencies.
-   `templates/`: The HTML templates for the web app.
-   `static/`: The CSS styles.
-   `vercel.json`: The configuration for Vercel deployment.

## 2. How It Works

The web app allows users to upload an image of a traffic sign. The Flask backend then preprocesses the image, feeds it into the pre-trained TensorFlow model, and displays the top 5 predictions to the user.

## 3. Deploying to Vercel

1.  **Sign up for Vercel:** Create an account at [vercel.com](https://vercel.com).
2.  **Install the Vercel CLI:**
    ```bash
    npm install -g vercel
    ```
3.  **Log in to your Vercel account:**
    ```bash
    vercel login
    ```
4.  **Deploy the application:**
    ```bash
    vercel --prod
    ```

Vercel will automatically detect the `vercel.json` file, install the dependencies from `requirements.txt`, and deploy the application.