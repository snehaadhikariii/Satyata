// src/api/satyata.js — Handles all communication with the backend
import axios from 'axios';

const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: BASE,
  headers: { 'Content-Type': 'application/json' },
  timeout: 20000,
});

function friendlyError(err) {
  if (err.code === 'ECONNABORTED') {
    return 'The request timed out. The backend may still be loading the model.';
  }
  if (err.response?.data?.error) {
    return err.response.data.error;
  }
  if (!err.response) {
    return 'Could not reach the backend at ' + BASE + '. Is "python manage.py runserver" running?';
  }
  return 'Something went wrong. Is the backend running?';
}

export const analyzeArticle = async ({ text, url }) => {
  try {
    const res = await api.post('/analyze', { text, url });
    return res.data;
  } catch (err) {
    throw new Error(friendlyError(err));
  }
};

export const getHistory = async () => {
  try {
    const res = await api.get('/history');
    return res.data;
  } catch (err) {
    throw new Error(friendlyError(err));
  }
};

export const checkHealth = async () => {
  try {
    const res = await api.get('/health');
    return res.data;
  } catch (err) {
    throw new Error(friendlyError(err));
  }
};