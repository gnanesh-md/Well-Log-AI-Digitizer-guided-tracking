// controllers/pointController.js
const Point = require('../models/pointsModel');
const fs = require('fs');
const fsPromises = fs.promises;
const path = require('path');
const { parse } = require('json2csv');

// Save points for a specific image
exports.savePoints = async (req, res) => {
  const { imageName, points } = req.body;

  try {
    const pointData = await Point.findOneAndUpdate(
      { imageName },
      { points },
      { upsert: true, new: true }
    );

    res.status(200).json({ message: 'Points saved successfully', data: pointData });
  } catch (error) {
    console.error('Error saving points:', error);
    res.status(500).json({ message: 'Error saving points' });
  }
};

exports.exportPoints = async (req, res) => {
  const imageName = req.params.imageName;
  let pointsGraph1 = req.body?.pointsGraph1 || req.query?.pointsGraph1;
  let pointsGraph2 = req.body?.pointsGraph2 || req.query?.pointsGraph2;

  // Fallback to database lookup if no points were passed in request body/query
  if (!pointsGraph1 && !pointsGraph2) {
    try {
      const savedPoint = await Point.findOne({ imageName });
      if (savedPoint && savedPoint.points) {
        pointsGraph1 = savedPoint.points;
        pointsGraph2 = [];
      }
    } catch (dbErr) {
      console.error('Database query error in exportPoints:', dbErr);
    }
  }

  // Validate input
  if (!Array.isArray(pointsGraph1) && !Array.isArray(pointsGraph2)) {
      return res.status(400).json({ error: 'Invalid or missing points data' });
  }

  pointsGraph1 = pointsGraph1 || [];
  pointsGraph2 = pointsGraph2 || [];

  // Create LAS data
  let lasData = 'X,Y,Z\n'; // Header
  const createLASData = (points) => {
      return points.map(point => `${point.x},${point.y},0`).join('\n');
  };

  if (pointsGraph1.length > 0) lasData += createLASData(pointsGraph1) + '\n';
  if (pointsGraph2.length > 0) lasData += createLASData(pointsGraph2) + '\n';

  // Create a filename
  const filename = `${imageName}.las`;
  const filePath = path.join(__dirname, '../uploads', filename);

  try {
      // Write the LAS data to a file
      await fsPromises.writeFile(filePath, lasData);
      
      // Send the file for download
      res.download(filePath, filename, async (err) => {
          if (err) {
              console.error('Error sending the file:', err);
              try {
                  await fsPromises.unlink(filePath);
              } catch (unlinkErr) {
                  console.error('Error deleting the file after failed download:', unlinkErr);
              }
              if (!res.headersSent) {
                  return res.status(500).json({ error: 'Error sending the file' });
              }
              return;
          }
          // Delete the file after sending it
          try {
              await fsPromises.unlink(filePath);
          } catch (unlinkErr) {
              console.error('Error deleting the file after download:', unlinkErr);
          }
      });
  } catch (err) {
      console.error('Error saving the file:', err);
      return res.status(500).json({ error: 'Error saving the file' });
  }
};

